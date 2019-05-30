#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "zip.h"
#include <omp.h>

char* zipfile = "MNIST_BIN.zip";
char* trainFileName = "mnist_train.bin";
char* testFileName = "mnist_test.bin";

// -------------------------------------------------------------------------------------------------------
// 환경 설정
int numOfCase = 60000;              // 학습 데이터 건수
int numOfTest = 10000;              // 학습 테스트 건수
int batchSize = 1000;               // 미니 배치 사이즈(1000)
int numOfXinputs = 784;             // 28 * 28 이미지
int numOfHiddenNodes = 100;         // 은닉층 노드 수
int numOfOutputNodes = 10;          // 숫자 0~9 판별
int numOfEpoch = 20;                // Epoch 횟수(20)
double learningRate = 0.01;         // 학습률(0.01)
// -------------------------------------------------------------------------------------------------------

const int MEAN_SQUARED_ERROR = 0;
const int CROSS_ENTROPY_ERROR = 1;
const int SIGMOID = 0;
const int SOFTMAX = 1;

double* Xdata;                      // 전체 데이터(입력)
int* Tdata;                         // 전체 데이터(목표치)
double* Xinput;                     // 입력 데이터
int* Tinput;                        // 목표치 데이터
double* Yhidden;                    // 은닉층 출력
double* Youtput;                    // 출력층 출력
double* Whidden;                    // 입력층과 은닉층 사이의 가중치 행렬
double* Bhidden;                    // 은닉층의 바이어스
double* Woutput;                    // 은닉층과 출력층 사이의 가중치 행렬
double* Boutput;                    // 출력층의 바이어스
double* DWhidden;                   // 입력층과 은닉층 사이의 가중치 갱신값 행렬
double* DBhidden;                   // 은닉층의 바이어스 갱신값
double* DWoutput;                   // 은닉층과 출력층 사이의 가중치 갱신값 행렬
double* DBoutput;                   // 출력층의 바이어스 갱신값
double* Dhidden;                    // 은닉층의 델타
double* Doutput;                    // 출력층의 델타

void allocateMemory(void);
void deallocateMemory(void);
void readData(char* fileName, int lines);
void samplingData(int size, int total);
void initializeWB(void);
double gaussianRandom(double average, double stdev);
void initialize(void);
void calculateXWB(double* Y, double* X, double* W, double* B, int m, int n, int o, int type);
double calculateErrorCost(int type);
void calculateDeltaOutput(int type);
void calculateDeltaHidden(void);
void updateWB(void);
void trainingSerial(void);
void trainingParallel(void);
void test(int records);

int main(int argc, char* argv[]) {
    
    /*
    FILE *fp;
    fopen_s(&fp, "cost.txt", "wt");
    if (fp == NULL) {
        printf("파일 열기 실패\n");
        exit(0);
    }
    */

    allocateMemory();                                                       // 메모리 할당                                                                     
    readData(trainFileName, numOfCase);                                     // 학습 데이터 읽기
    
    clock_t begin = clock();                                                // 실행 시간 측정

    initializeWB();                                                         // W, B를 정규분포 값으로 초기화
    double totalErrorCost = 0.0;

    int iteration = numOfCase / batchSize;                                  // 1회 Epoch를 수행 할 때까지의 반복 횟수
    for (int i = 0; i < numOfEpoch; i++) {                                  // Epoch 반복
        totalErrorCost = 0.0;
        for (int j = 0; j < iteration; j++) {
            samplingData(batchSize, numOfCase);                             // 전체 학습 데이터에서 배치 사이즈만큼 랜덤 선택
            initialize();                                                   // 초기화 작업
            
            /* 순차 처리 방식. 1차 코딩
            // 순전파
            // 은닉층 출력 계산. Yhidden = f(Xinputs * Whidden + Bhidden)
            calculateXWB(Yhidden, Xinput, Whidden, Bhidden, batchSize, numOfXinputs, numOfHiddenNodes, SIGMOID); 
            // 출력층 출력 계산. Youtput = f(Yhidden * Woutput + Boutput);
            calculateXWB(Youtput, Yhidden, Woutput, Boutput, batchSize, numOfHiddenNodes, numOfOutputNodes, SOFTMAX);

            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < numOfOutputNodes; j++) {
                    printf("%10.3lf ", Youtput[i * numOfOutputNodes + j]);
                }
            }
            
            // 오차 비용 계산. 
            double t = calculateErrorCost(CROSS_ENTROPY_ERROR); 
            totalErrorCost += t;
            printf("[%d, %d] 오차 비용 = %lf\n", i, j, t);

            // 오차역전파
            calculateDeltaOutput(SOFTMAX);                                  // 출력층 델타 계산. 
            calculateDeltaHidden();                                         // 은닉층 델타 계산.           
            updateWB();                                                     // 가중치 및 바이어스 갱신
            */

            printf("[%d, %d] ", i, j);
            //trainingSerial();                                             // 순차 처리 방식. 2차 재코딩 
            trainingParallel();                                             // 병렬 처리 방식. OpenMP 적용
        }
        //fprintf(fp, "%lf\n", totalErrorCost / iteration);
    }
    //fclose(fp);

    clock_t end = clock();                                                  // 실행 시간 측정
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("수행 시간 : %lf 초\n", time_spent);

    test(numOfCase);                                                        // 학습 데이터로 추정 결과 확인

    readData(testFileName, numOfTest);                                      // 테스트 데이터로 추정 결과 확인
    test(numOfTest);

    deallocateMemory();                                                     // 메모리 해제
    return 0;
}

void test(int records) {
    // 배치 사이즈로 분할
    int iteration = records / batchSize;
    int index = 0;
    int success = 0;
    int found = 0;
    for (int i = 0; i < iteration; i++) {
        index = 0;
        for (int j = i * batchSize; j < (i + 1) * batchSize; j++) {
            Tinput[index] = Tdata[j];
            for (int k = 0; k < numOfXinputs; k++) {
                Xinput[index * numOfXinputs + k] = Xdata[j * numOfXinputs + k];
            }
            index++;
        }
        
        calculateXWB(Yhidden, Xinput, Whidden, Bhidden, batchSize, numOfXinputs, numOfHiddenNodes, SIGMOID);
        calculateXWB(Youtput, Yhidden, Woutput, Boutput, batchSize, numOfHiddenNodes, numOfOutputNodes, SOFTMAX);

        for (int j = 0; j < batchSize; j++) {
            double max = Youtput[j * numOfOutputNodes];
            found = 0;
            for (int k = 1; k < numOfOutputNodes; k++) {
                if (max < Youtput[j * numOfOutputNodes + k]) {
                    max = Youtput[j * numOfOutputNodes + k];
                    found = k;
                }
            }
            if (Tinput[j] == found) success++;
        }
    }

    printf("\n입력 데이터 갯수 : %d\n", records);
    printf("성공 : %d, 실패 : %d\n", success, records - success);
    printf("성공률 : %6.2lf%%\n", (double)success / records * 100.0);
}


void updateWB() {
    // 입력층-은닉층 가중치 갱신
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] -= learningRate * DWhidden[i * numOfHiddenNodes + j];
        }
    }

    // 은닉층 바이어스 및 은닉층-출력층 가중치 갱신
    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] -= learningRate * DBhidden[i];
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] -= learningRate * DWoutput[i * numOfOutputNodes + j];
        }
    }

    // 출력층 바이어스 갱신
    for (int i = 0; i < numOfOutputNodes; i++) {
        Boutput[i] -= learningRate * DBoutput[i];
    }
}

void calculateDeltaHidden(void) {
    // Dhidden = Yhidden * (1 - Yhidden) * Doutput * tr(Woutput)
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfOutputNodes; k++) {
                dot += Doutput[i * numOfOutputNodes + k] * Woutput[j * numOfOutputNodes + k];
            }
            Dhidden[i * numOfHiddenNodes + j] = dot * Yhidden[i * numOfHiddenNodes + j]
                * (1 - Yhidden[i * numOfHiddenNodes + j]);
            // 델타 값은 바이어스의 갱신값과 동일
            DBhidden[j] += Dhidden[i * numOfHiddenNodes + j];
        }
    }
    // 입력층과 은닉층 간의 가중치 갱신값 계산
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numOfXinputs; j++) {
            for (int k = 0; k < numOfHiddenNodes; k++) {
                DWhidden[j * numOfHiddenNodes + k] += Xinput[i * numOfXinputs + j] * Dhidden[i * numOfHiddenNodes + k];
            }
        }
    }
}

void calculateDeltaOutput(int type) {
    // function type : SIGMOID - 출력층에서 시그모이드 함수와 오차 제곱합 비용 함수를 사용한 경우
    //                           Doutput = Youtput * (1 - Youtput) * (Youtput - Tinput)
    //                 SOFTMAX - 출력층에서 소프트맥스 함수와 교차 엔트로피 비용 함수를 사용한 경우
    //                     Doutput = Youtput - Tinput
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numOfOutputNodes; j++) {
            double error = 0.0;
            if (Tinput[i] == j) {
                error = Youtput[i * numOfOutputNodes + j] - 1;
            }
            else {
                error = Youtput[i * numOfOutputNodes + j];
            }
            if (type == SIGMOID) {
                Doutput[i * numOfOutputNodes + j] = Youtput[i * numOfOutputNodes + j]
                    * (1 - Youtput[i * numOfOutputNodes + j]) * error;
            }
            else {
                Doutput[i * numOfOutputNodes + j] = error;
            }
            // 델타 값은 바이어스의 갱신값과 동일
            DBoutput[j] += Doutput[i * numOfOutputNodes + j];
        }
    }
    // 은닉층과 출력층 간의 가중치 갱신값 계산
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            for (int k = 0; k < numOfOutputNodes; k++) {
                DWoutput[j * numOfOutputNodes + k] += Yhidden[i * numOfHiddenNodes + j] * Doutput[i * numOfOutputNodes + k];
            }
        }
    }
}

double calculateErrorCost(int type) {
    // 오차 비용 계산
    // function type : MEAN_SQUARED_ERROR - 0.5 * 오차제곱 합, CROSS_ENTROPY_ERROR - 교차 엔트로피 함수
    double totalCost = 0.0;
    for (int i = 0; i < batchSize; i++) {
        double errorCost = 0.0;
        for (int j = 0; j < numOfOutputNodes; j++) {
            if (Tinput[i] == j) {
                if (type == MEAN_SQUARED_ERROR) {
                    errorCost += 0.5 * pow((1 - Youtput[i * numOfOutputNodes + j]), 2.0);   // 오차 제곱 합 계산
                }
                else {
                    errorCost += -1 * log(Youtput[i * numOfOutputNodes + j] + 0.0000001);   // 교차 엔트로피 계산
                }
            }
            else {
                if (type == MEAN_SQUARED_ERROR) {
                    errorCost += 0.5 * pow(-1 * Youtput[i * numOfOutputNodes + j], 2.0);    // 오차 제곱 합 계산
                }
            }
        }
        totalCost += errorCost;
    }
    totalCost = totalCost / batchSize;

    return totalCost;
}

void trainingSerial(void) {

    // Whidden 행렬 접근시 캐시 탈락을 방지하기 위해 전치행렬로 변환
    double* Thidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Thidden[j * numOfXinputs + i] = Whidden[i * numOfHiddenNodes + j];
        }
    }

    // Woutput 행렬 접근시 캐시 탈락을 방지하기 위해 전치행렬로 변환
    double* Toutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfOutputNodes; j++) {
            Toutput[j * numOfHiddenNodes + i] = Woutput[i * numOfOutputNodes + j];
        }
    }

    double totalCost = 0.0;
    
    for (int i = 0; i < batchSize; i++) {
        // 은닉층 출력 계산. 활성화 함수 : 시그모이드 함수
        for (int j = 0; j < numOfHiddenNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfXinputs; k++) {
                dot += Xinput[i * numOfXinputs + k] * Thidden[j * numOfXinputs + k];
            }
            Yhidden[i * numOfHiddenNodes + j] = 1.0 / (1.0 + exp(-1.0 * (dot + Bhidden[j])));       // 시그모이드 함수값 계산
        }

        // 출력층 출력 계산. 활성화 함수 : 소프트맥스 함수
        double total = 0.0;
        double max = DBL_MIN;

        for (int j = 0; j < numOfOutputNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfHiddenNodes; k++) {
                dot += Yhidden[i * numOfHiddenNodes + k] * Toutput[j * numOfHiddenNodes + k];
            }
            Youtput[i * numOfOutputNodes + j] = dot + Boutput[j];
            if (Youtput[i * numOfOutputNodes + j] > max) 
                max = Youtput[i * numOfOutputNodes + j];                                            // 최대값 찾기     
        }

        for (int j = 0; j < numOfOutputNodes; j++) {
            total += exp(Youtput[i * numOfOutputNodes + j] - max);                                  // 소프트맥스 함수의 분모 계산
        }

        for (int j = 0; j < numOfOutputNodes; j++) {
            Youtput[i * numOfOutputNodes + j] = exp(Youtput[i * numOfOutputNodes + j] - max) / total;     // 소프트맥스 함수값 계산
        }

        // 오차계산
        double errorCost = 0.0;
        for (int j = 0; j < numOfOutputNodes; j++) {
            if (Tinput[i] == j) {
                    errorCost += -1 * log(Youtput[i * numOfOutputNodes + j] + 0.0000001);           // 교차 엔트로피 계산
            }
        }
        totalCost += errorCost;         // race condition
        
        // 출력층 델타 계산. 활성화 함수 : 소프트맥스 함수
        // 출력층에서 소프트맥스 함수와 교차 엔트로피 비용 함수를 사용한 경우
        // Doutput = Youtput - Tinput
        for (int j = 0; j < numOfOutputNodes; j++) {
            double error = 0.0;
            if (Tinput[i] == j) {
                error = Youtput[i * numOfOutputNodes + j] - 1;
            }
            else {
                error = Youtput[i * numOfOutputNodes + j];
            }
            Doutput[i * numOfOutputNodes + j] = error;
            DBoutput[j] += Doutput[i * numOfOutputNodes + j];       // 델타 값은 바이어스의 갱신값과 동일. 
        }

        // 은닉층과 출력층 간의 가중치 갱신값 계산
        for (int j = 0; j < numOfHiddenNodes; j++) {
            for (int k = 0; k < numOfOutputNodes; k++) {
                DWoutput[j * numOfOutputNodes + k] += Yhidden[i * numOfHiddenNodes + j] * Doutput[i * numOfOutputNodes + k];    
            }
        }

        // 은닉층 델타 계산. 활성화 함수 : 시그모이드 함수
        // Dhidden = Yhidden * (1 - Yhidden) * Doutput * tr(Woutput)
        for (int j = 0; j < numOfHiddenNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfOutputNodes; k++) {
                dot += Doutput[i * numOfOutputNodes + k] * Woutput[j * numOfOutputNodes + k];
            }
            Dhidden[i * numOfHiddenNodes + j] = dot * Yhidden[i * numOfHiddenNodes + j] * (1 - Yhidden[i * numOfHiddenNodes + j]);  // 델타 값은 바이어스의 갱신값과 동일           
            DBhidden[j] += Dhidden[i * numOfHiddenNodes + j];       // race condition
        }

        // 입력층과 은닉층 간의 가중치 갱신값 계산
        for (int j = 0; j < numOfXinputs; j++) {
            for (int k = 0; k < numOfHiddenNodes; k++) {
                DWhidden[j * numOfHiddenNodes + k] += Xinput[i * numOfXinputs + j] * Dhidden[i * numOfHiddenNodes + k];     
            }
        }
    }

    free(Thidden);
    free(Toutput);

    // 입력층-은닉층 가중치 갱신
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] -= learningRate * DWhidden[i * numOfHiddenNodes + j];
        }
    }

    // 은닉층 바이어스 및 은닉층-출력층 가중치 갱신
    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] -= learningRate * DBhidden[i];
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] -= learningRate * DWoutput[i * numOfOutputNodes + j];
        }
    }

    // 출력층 바이어스 갱신
    for (int i = 0; i < numOfOutputNodes; i++) {
        Boutput[i] -= learningRate * DBoutput[i];
    }

    totalCost = totalCost / batchSize;
    printf("오차 비용 = %lf\n", totalCost);

}

void trainingParallel(void) {

    // Whidden 행렬 접근시 캐시 탈락을 방지하기 위해 전치행렬로 변환
    double* Thidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Thidden[j * numOfXinputs + i] = Whidden[i * numOfHiddenNodes + j];
        }
    }

    // Woutput 행렬 접근시 캐시 탈락을 방지하기 위해 전치행렬로 변환
    double* Toutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfOutputNodes; j++) {
            Toutput[j * numOfHiddenNodes + i] = Woutput[i * numOfOutputNodes + j];
        }
    }

    double totalCost = 0.0;
 
    // 병렬 처리가 시작되는 곳
#pragma omp parallel num_threads(4)
    {
        // 스레드 로컬 변수 선언
        double localCost = 0.0;
        double * localDBoutput = (double *)calloc(numOfOutputNodes, sizeof(double));
        double * localDWoutput = (double *)calloc(numOfHiddenNodes * numOfOutputNodes, sizeof(double));
        double * localDBhidden = (double *)calloc(numOfHiddenNodes, sizeof(double));
        double * localDWhidden = (double *)calloc(numOfXinputs * numOfHiddenNodes, sizeof(double));
        int i = 0;

#pragma omp for
        for (i = 0; i < batchSize; i++) {
            // 은닉층 출력 계산. 활성화 함수 : 시그모이드 함수
            for (int j = 0; j < numOfHiddenNodes; j++) {
                double dot = 0.0;
                for (int k = 0; k < numOfXinputs; k++) {
                    dot += Xinput[i * numOfXinputs + k] * Thidden[j * numOfXinputs + k];
                }
                Yhidden[i * numOfHiddenNodes + j] = 1.0 / (1.0 + exp(-1.0 * (dot + Bhidden[j])));       // 시그모이드 함수값 계산
            }

            // 출력층 출력 계산. 활성화 함수 : 소프트맥스 함수
            double total = 0.0;
            double max = DBL_MIN;

            for (int j = 0; j < numOfOutputNodes; j++) {
                double dot = 0.0;
                for (int k = 0; k < numOfHiddenNodes; k++) {
                    dot += Yhidden[i * numOfHiddenNodes + k] * Toutput[j * numOfHiddenNodes + k];
                }
                Youtput[i * numOfOutputNodes + j] = dot + Boutput[j];
                if (Youtput[i * numOfOutputNodes + j] > max)
                    max = Youtput[i * numOfOutputNodes + j];                                            // 최대값 찾기  
            }

            for (int j = 0; j < numOfOutputNodes; j++) {
                total += exp(Youtput[i * numOfOutputNodes + j] - max);                                  // 소프트맥스 함수의 분모 계산
            }

            for (int j = 0; j < numOfOutputNodes; j++) {
                Youtput[i * numOfOutputNodes + j] = exp(Youtput[i * numOfOutputNodes + j] - max) / total;     // 소프트맥스 함수값 계산
            }

            // 오차계산
            double errorCost = 0.0;
            for (int j = 0; j < numOfOutputNodes; j++) {
                if (Tinput[i] == j) {
                    errorCost += -1 * log(Youtput[i * numOfOutputNodes + j] + 0.0000001);           // 교차 엔트로피 계산
                }
            }
            localCost += errorCost;         // race condition

            // 출력층 델타 계산. 활성화 함수 : 소프트맥스 함수
            // 출력층에서 소프트맥스 함수와 교차 엔트로피 비용 함수를 사용한 경우
            // Doutput = Youtput - Tinput
            for (int j = 0; j < numOfOutputNodes; j++) {
                double error = 0.0;
                if (Tinput[i] == j) {
                    error = Youtput[i * numOfOutputNodes + j] - 1;
                }
                else {
                    error = Youtput[i * numOfOutputNodes + j];
                }
                Doutput[i * numOfOutputNodes + j] = error;
                localDBoutput[j] += Doutput[i * numOfOutputNodes + j];       // 델타 값은 바이어스의 갱신값과 동일. race condition
            }

            // 은닉층과 출력층 간의 가중치 갱신값 계산
            for (int j = 0; j < numOfHiddenNodes; j++) {
                for (int k = 0; k < numOfOutputNodes; k++) {
                    localDWoutput[j * numOfOutputNodes + k] += Yhidden[i * numOfHiddenNodes + j] * Doutput[i * numOfOutputNodes + k];    // race condition
                }
            }

            // 은닉층 델타 계산. 활성화 함수 : 시그모이드 함수
            // Dhidden = Yhidden * (1 - Yhidden) * Doutput * tr(Woutput)
            for (int j = 0; j < numOfHiddenNodes; j++) {
                double dot = 0.0;
                for (int k = 0; k < numOfOutputNodes; k++) {
                    dot += Doutput[i * numOfOutputNodes + k] * Woutput[j * numOfOutputNodes + k];
                }
                Dhidden[i * numOfHiddenNodes + j] = dot * Yhidden[i * numOfHiddenNodes + j] * (1 - Yhidden[i * numOfHiddenNodes + j]);  // 델타 값은 바이어스의 갱신값과 동일           
                localDBhidden[j] += Dhidden[i * numOfHiddenNodes + j];       // race condition
            }

            // 입력층과 은닉층 간의 가중치 갱신값 계산
            for (int j = 0; j < numOfXinputs; j++) {
                for (int k = 0; k < numOfHiddenNodes; k++) {
                    localDWhidden[j * numOfHiddenNodes + k] += Xinput[i * numOfXinputs + j] * Dhidden[i * numOfHiddenNodes + k];     // race condition
                }
            }
        }   // for 문 종료
#pragma omp critical
        {
            totalCost += localCost;
            for (int i = 0; i < numOfOutputNodes; i++) {
                DBoutput[i] += localDBoutput[i];

            }
            for (int i = 0; i < numOfHiddenNodes; i++) {
                for (int j = 0; j < numOfOutputNodes; j++) {
                    DWoutput[i * numOfOutputNodes + j] += localDWoutput[i * numOfOutputNodes + j];
                }
            }

            for (int i = 0; i < numOfHiddenNodes; i++) {
                DBhidden[i] += localDBhidden[i];
            }

            for (int i = 0; i < numOfXinputs; i++) {
                for (int j = 0; j < numOfHiddenNodes; j++) {
                    DWhidden[i * numOfHiddenNodes + j] += localDWhidden[i * numOfHiddenNodes + j];
                }
            }
        }

        free(localDBoutput);
        free(localDWoutput);
        free(localDBhidden);
        free(localDWhidden);
    }

    free(Thidden);
    free(Toutput);

    // 입력층-은닉층 가중치 갱신
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] -= learningRate * DWhidden[i * numOfHiddenNodes + j];
        }
    }

    // 은닉층 바이어스 및 은닉층-출력층 가중치 갱신
    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] -= learningRate * DBhidden[i];
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] -= learningRate * DWoutput[i * numOfOutputNodes + j];
        }
    }

    // 출력층 바이어스 갱신
    for (int i = 0; i < numOfOutputNodes; i++) {
        Boutput[i] -= learningRate * DBoutput[i];
    }

    totalCost = totalCost / batchSize;
    printf("오차 비용 = %lf\n", totalCost);

}

void calculateXWB(double* Y, double* X, double* W, double* B, int m, int n, int o, int type) {
    // Y = function(XW + B)
    // X : m * n 행렬
    // W : n * o 행렬
    // B : 1 * o 행렬
    // Y : m * o 행렬
    // function type : SIGMOID - sigmoid 함수, SOFTMAX - softmax 함수 적용
    // W 행렬 접근시 캐시 탈락을 방지하기 위해 전치행렬로 변환
    if (type != SIGMOID && type != SOFTMAX) type = SIGMOID;   

    double* T = (double *)malloc(sizeof(double) * n * o);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < o; j++) {
            T[j * n + i] = W[i * o + j];
        }
    }

    for (int i = 0; i < m; i++) {
        double total = 0.0;
        for (int j = 0; j < o; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++) {
                dot += X[i * n + k] * T[j * n + k];
            }
            if (type == SIGMOID) {
                Y[i * o + j] = 1.0 / (1.0 + exp(-1.0 * (dot + B[j])));  // 시그모이드 함수값 계산
            }
            else {
                Y[i * o + j] = dot + B[j];
                total += exp(Y[i * o + j]);                             // 소프트맥스 함수의 분모 계산
            }    
        }
        if (type == SOFTMAX) {
            for (int j = 0; j < o; j++) {
                Y[i * o + j] = exp(Y[i * o + j]) / total;               // 소프트맥스 함수값 계산
            }
        }
    }
    free(T);
}

void initialize(void) {
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            DWhidden[i * numOfHiddenNodes + j] = 0.0;
        }
    }

    for (int i = 0; i < numOfHiddenNodes; i++) {
        DBhidden[i] = 0.0;
        for (int j = 0; j < numOfOutputNodes; j++) {
            DWoutput[i * numOfOutputNodes + j] = 0.0;
        }
    }
    
    for (int i = 0; i < numOfOutputNodes; i++) {
        DBoutput[i] = 0.0;
    }
}

void initializeWB(void) {
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] = gaussianRandom(0, 1);
        }
    }

    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] = gaussianRandom(0, 1);
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] = gaussianRandom(0, 1);
        }
    }

    for (int i = 0; i < numOfOutputNodes; i++) {
        Boutput[i] = gaussianRandom(0, 1);
    }
}

void samplingData(int size, int total) {
    for (int i = 0; i < size; i++) {
        int selected = (size == total)? i : (int)(total * ((double)rand() / (RAND_MAX + 1)));
        Tinput[i] = Tdata[selected];
        for (int j = 0; j < numOfXinputs; j++) {
            Xinput[i * numOfXinputs + j] = Xdata[selected * numOfXinputs + j];
        }
    }
}

void allocateMemory(void) {
    Xdata = (double *)malloc(sizeof(double) * numOfCase * numOfXinputs);
    if (Xdata == NULL) {
        printf("Xdata 메모리 할당 실패");
        exit(1);
    }

    Tdata = (int *)malloc(sizeof(int) * numOfCase);
    if (Tdata == NULL) {
        printf("Tdata 메모리 할당 실패");
        exit(1);
    }

    Xinput = (double *)malloc(sizeof(double) * batchSize * numOfXinputs);
    if (Xinput == NULL) {
        printf("Xdata 메모리 할당 실패");
        exit(1);
    }

    Tinput = (int *)malloc(sizeof(int) * batchSize);
    if (Tinput == NULL) {
        printf("Tinput 메모리 할당 실패");
        exit(1);
    }

    Yhidden = (double *)malloc(sizeof(double) * batchSize * numOfHiddenNodes);
    if (Yhidden == NULL) {
        printf("Yhidden 메모리 할당 실패");
        exit(1);
    }

    Youtput = (double *)malloc(sizeof(double) * batchSize * numOfOutputNodes);
    if (Youtput == NULL) {
        printf("Youtput 메모리 할당 실패");
        exit(1);
    }

    Whidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    if (Whidden == NULL) {
        printf("Whidden 메모리 할당 실패");
        exit(1);
    }

    Bhidden = (double *)malloc(sizeof(double) * numOfHiddenNodes);
    if (Bhidden == NULL) {
        printf("Bhidden 메모리 할당 실패");
        exit(1);
    }

    Woutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    if (Woutput == NULL) {
        printf("Woutput 메모리 할당 실패");
        exit(1);
    }

    Boutput = (double *)malloc(sizeof(double) * numOfOutputNodes);
    if (Boutput == NULL) {
        printf("Boutput 메모리 할당 실패");
        exit(1);
    }

    DWhidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    if (DWhidden == NULL) {
        printf("DWhidden 메모리 할당 실패");
        exit(1);
    }

    DBhidden = (double *)malloc(sizeof(double) * numOfHiddenNodes);
    if (DBhidden == NULL) {
        printf("DBhidden 메모리 할당 실패");
        exit(1);
    }

    DWoutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    if (DWoutput == NULL) {
        printf("DWoutput 메모리 할당 실패");
        exit(1);
    }

    DBoutput = (double *)malloc(sizeof(double) * numOfOutputNodes);
    if (DBoutput == NULL) {
        printf("DBoutput 메모리 할당 실패");
        exit(1);
    }

    Dhidden = (double *)malloc(sizeof(double) * batchSize * numOfHiddenNodes);
    if (Dhidden == NULL) {
        printf("Dhidden 메모리 할당 실패");
        exit(1);
    }

    Doutput = (double *)malloc(sizeof(double) * batchSize * numOfOutputNodes);
    if (Doutput == NULL) {
        printf("Doutput 메모리 할당 실패");
        exit(1);
    }
}

void deallocateMemory(void) {
    free(Xdata);
    free(Tdata);
    free(Xinput);
    free(Tinput);
    free(Yhidden);
    free(Youtput);
    free(Whidden);
    free(Bhidden);
    free(Woutput);
    free(Boutput);
    free(DWhidden);
    free(DBhidden);
    free(DWoutput);
    free(DBoutput);
    free(Dhidden);
    free(Doutput);
}

void readData(char* fileName, int lines) {
    void *buf = NULL;
    size_t bufsize;

    struct zip_t *zip = zip_open(zipfile, 0, 'r');              // 압축 파일 읽기
    zip_entry_open(zip, fileName);                              // 지정된 파일 추출 후, 메모리로 로드
    zip_entry_read(zip, &buf, &bufsize);
    zip_entry_close(zip);
    unsigned char *arr = (unsigned char *)buf;

    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < numOfXinputs + 1; j++) {
            if (j == 0) {
                Tdata[i] = (int)arr[i * (numOfXinputs + 1)];
            }
            else {
                Xdata[i * numOfXinputs + j - 1] = (double)arr[i * (numOfXinputs + 1) + j] / 255.0;
            }
        }
    }
    free(buf);
}

double gaussianRandom(double average, double stdev) {
    double v1, v2, s, temp;

    do {
        v1 = 2 * ((double)rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        v2 = 2 * ((double)rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = sqrt((-2 * log(s)) / s);

    temp = v1 * s;
    temp = (stdev * temp) + average;
    
    return temp;
}