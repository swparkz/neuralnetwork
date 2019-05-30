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
// ȯ�� ����
int numOfCase = 60000;              // �н� ������ �Ǽ�
int numOfTest = 10000;              // �н� �׽�Ʈ �Ǽ�
int batchSize = 1000;               // �̴� ��ġ ������(1000)
int numOfXinputs = 784;             // 28 * 28 �̹���
int numOfHiddenNodes = 100;         // ������ ��� ��
int numOfOutputNodes = 10;          // ���� 0~9 �Ǻ�
int numOfEpoch = 20;                // Epoch Ƚ��(20)
double learningRate = 0.01;         // �н���(0.01)
// -------------------------------------------------------------------------------------------------------

const int MEAN_SQUARED_ERROR = 0;
const int CROSS_ENTROPY_ERROR = 1;
const int SIGMOID = 0;
const int SOFTMAX = 1;

double* Xdata;                      // ��ü ������(�Է�)
int* Tdata;                         // ��ü ������(��ǥġ)
double* Xinput;                     // �Է� ������
int* Tinput;                        // ��ǥġ ������
double* Yhidden;                    // ������ ���
double* Youtput;                    // ����� ���
double* Whidden;                    // �Է����� ������ ������ ����ġ ���
double* Bhidden;                    // �������� ���̾
double* Woutput;                    // �������� ����� ������ ����ġ ���
double* Boutput;                    // ������� ���̾
double* DWhidden;                   // �Է����� ������ ������ ����ġ ���Ű� ���
double* DBhidden;                   // �������� ���̾ ���Ű�
double* DWoutput;                   // �������� ����� ������ ����ġ ���Ű� ���
double* DBoutput;                   // ������� ���̾ ���Ű�
double* Dhidden;                    // �������� ��Ÿ
double* Doutput;                    // ������� ��Ÿ

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
        printf("���� ���� ����\n");
        exit(0);
    }
    */

    allocateMemory();                                                       // �޸� �Ҵ�                                                                     
    readData(trainFileName, numOfCase);                                     // �н� ������ �б�
    
    clock_t begin = clock();                                                // ���� �ð� ����

    initializeWB();                                                         // W, B�� ���Ժ��� ������ �ʱ�ȭ
    double totalErrorCost = 0.0;

    int iteration = numOfCase / batchSize;                                  // 1ȸ Epoch�� ���� �� �������� �ݺ� Ƚ��
    for (int i = 0; i < numOfEpoch; i++) {                                  // Epoch �ݺ�
        totalErrorCost = 0.0;
        for (int j = 0; j < iteration; j++) {
            samplingData(batchSize, numOfCase);                             // ��ü �н� �����Ϳ��� ��ġ �����ŭ ���� ����
            initialize();                                                   // �ʱ�ȭ �۾�
            
            /* ���� ó�� ���. 1�� �ڵ�
            // ������
            // ������ ��� ���. Yhidden = f(Xinputs * Whidden + Bhidden)
            calculateXWB(Yhidden, Xinput, Whidden, Bhidden, batchSize, numOfXinputs, numOfHiddenNodes, SIGMOID); 
            // ����� ��� ���. Youtput = f(Yhidden * Woutput + Boutput);
            calculateXWB(Youtput, Yhidden, Woutput, Boutput, batchSize, numOfHiddenNodes, numOfOutputNodes, SOFTMAX);

            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < numOfOutputNodes; j++) {
                    printf("%10.3lf ", Youtput[i * numOfOutputNodes + j]);
                }
            }
            
            // ���� ��� ���. 
            double t = calculateErrorCost(CROSS_ENTROPY_ERROR); 
            totalErrorCost += t;
            printf("[%d, %d] ���� ��� = %lf\n", i, j, t);

            // ����������
            calculateDeltaOutput(SOFTMAX);                                  // ����� ��Ÿ ���. 
            calculateDeltaHidden();                                         // ������ ��Ÿ ���.           
            updateWB();                                                     // ����ġ �� ���̾ ����
            */

            printf("[%d, %d] ", i, j);
            //trainingSerial();                                             // ���� ó�� ���. 2�� ���ڵ� 
            trainingParallel();                                             // ���� ó�� ���. OpenMP ����
        }
        //fprintf(fp, "%lf\n", totalErrorCost / iteration);
    }
    //fclose(fp);

    clock_t end = clock();                                                  // ���� �ð� ����
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("���� �ð� : %lf ��\n", time_spent);

    test(numOfCase);                                                        // �н� �����ͷ� ���� ��� Ȯ��

    readData(testFileName, numOfTest);                                      // �׽�Ʈ �����ͷ� ���� ��� Ȯ��
    test(numOfTest);

    deallocateMemory();                                                     // �޸� ����
    return 0;
}

void test(int records) {
    // ��ġ ������� ����
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

    printf("\n�Է� ������ ���� : %d\n", records);
    printf("���� : %d, ���� : %d\n", success, records - success);
    printf("������ : %6.2lf%%\n", (double)success / records * 100.0);
}


void updateWB() {
    // �Է���-������ ����ġ ����
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] -= learningRate * DWhidden[i * numOfHiddenNodes + j];
        }
    }

    // ������ ���̾ �� ������-����� ����ġ ����
    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] -= learningRate * DBhidden[i];
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] -= learningRate * DWoutput[i * numOfOutputNodes + j];
        }
    }

    // ����� ���̾ ����
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
            // ��Ÿ ���� ���̾�� ���Ű��� ����
            DBhidden[j] += Dhidden[i * numOfHiddenNodes + j];
        }
    }
    // �Է����� ������ ���� ����ġ ���Ű� ���
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numOfXinputs; j++) {
            for (int k = 0; k < numOfHiddenNodes; k++) {
                DWhidden[j * numOfHiddenNodes + k] += Xinput[i * numOfXinputs + j] * Dhidden[i * numOfHiddenNodes + k];
            }
        }
    }
}

void calculateDeltaOutput(int type) {
    // function type : SIGMOID - ��������� �ñ׸��̵� �Լ��� ���� ������ ��� �Լ��� ����� ���
    //                           Doutput = Youtput * (1 - Youtput) * (Youtput - Tinput)
    //                 SOFTMAX - ��������� ����Ʈ�ƽ� �Լ��� ���� ��Ʈ���� ��� �Լ��� ����� ���
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
            // ��Ÿ ���� ���̾�� ���Ű��� ����
            DBoutput[j] += Doutput[i * numOfOutputNodes + j];
        }
    }
    // �������� ����� ���� ����ġ ���Ű� ���
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            for (int k = 0; k < numOfOutputNodes; k++) {
                DWoutput[j * numOfOutputNodes + k] += Yhidden[i * numOfHiddenNodes + j] * Doutput[i * numOfOutputNodes + k];
            }
        }
    }
}

double calculateErrorCost(int type) {
    // ���� ��� ���
    // function type : MEAN_SQUARED_ERROR - 0.5 * �������� ��, CROSS_ENTROPY_ERROR - ���� ��Ʈ���� �Լ�
    double totalCost = 0.0;
    for (int i = 0; i < batchSize; i++) {
        double errorCost = 0.0;
        for (int j = 0; j < numOfOutputNodes; j++) {
            if (Tinput[i] == j) {
                if (type == MEAN_SQUARED_ERROR) {
                    errorCost += 0.5 * pow((1 - Youtput[i * numOfOutputNodes + j]), 2.0);   // ���� ���� �� ���
                }
                else {
                    errorCost += -1 * log(Youtput[i * numOfOutputNodes + j] + 0.0000001);   // ���� ��Ʈ���� ���
                }
            }
            else {
                if (type == MEAN_SQUARED_ERROR) {
                    errorCost += 0.5 * pow(-1 * Youtput[i * numOfOutputNodes + j], 2.0);    // ���� ���� �� ���
                }
            }
        }
        totalCost += errorCost;
    }
    totalCost = totalCost / batchSize;

    return totalCost;
}

void trainingSerial(void) {

    // Whidden ��� ���ٽ� ĳ�� Ż���� �����ϱ� ���� ��ġ��ķ� ��ȯ
    double* Thidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Thidden[j * numOfXinputs + i] = Whidden[i * numOfHiddenNodes + j];
        }
    }

    // Woutput ��� ���ٽ� ĳ�� Ż���� �����ϱ� ���� ��ġ��ķ� ��ȯ
    double* Toutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfOutputNodes; j++) {
            Toutput[j * numOfHiddenNodes + i] = Woutput[i * numOfOutputNodes + j];
        }
    }

    double totalCost = 0.0;
    
    for (int i = 0; i < batchSize; i++) {
        // ������ ��� ���. Ȱ��ȭ �Լ� : �ñ׸��̵� �Լ�
        for (int j = 0; j < numOfHiddenNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfXinputs; k++) {
                dot += Xinput[i * numOfXinputs + k] * Thidden[j * numOfXinputs + k];
            }
            Yhidden[i * numOfHiddenNodes + j] = 1.0 / (1.0 + exp(-1.0 * (dot + Bhidden[j])));       // �ñ׸��̵� �Լ��� ���
        }

        // ����� ��� ���. Ȱ��ȭ �Լ� : ����Ʈ�ƽ� �Լ�
        double total = 0.0;
        double max = DBL_MIN;

        for (int j = 0; j < numOfOutputNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfHiddenNodes; k++) {
                dot += Yhidden[i * numOfHiddenNodes + k] * Toutput[j * numOfHiddenNodes + k];
            }
            Youtput[i * numOfOutputNodes + j] = dot + Boutput[j];
            if (Youtput[i * numOfOutputNodes + j] > max) 
                max = Youtput[i * numOfOutputNodes + j];                                            // �ִ밪 ã��     
        }

        for (int j = 0; j < numOfOutputNodes; j++) {
            total += exp(Youtput[i * numOfOutputNodes + j] - max);                                  // ����Ʈ�ƽ� �Լ��� �и� ���
        }

        for (int j = 0; j < numOfOutputNodes; j++) {
            Youtput[i * numOfOutputNodes + j] = exp(Youtput[i * numOfOutputNodes + j] - max) / total;     // ����Ʈ�ƽ� �Լ��� ���
        }

        // �������
        double errorCost = 0.0;
        for (int j = 0; j < numOfOutputNodes; j++) {
            if (Tinput[i] == j) {
                    errorCost += -1 * log(Youtput[i * numOfOutputNodes + j] + 0.0000001);           // ���� ��Ʈ���� ���
            }
        }
        totalCost += errorCost;         // race condition
        
        // ����� ��Ÿ ���. Ȱ��ȭ �Լ� : ����Ʈ�ƽ� �Լ�
        // ��������� ����Ʈ�ƽ� �Լ��� ���� ��Ʈ���� ��� �Լ��� ����� ���
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
            DBoutput[j] += Doutput[i * numOfOutputNodes + j];       // ��Ÿ ���� ���̾�� ���Ű��� ����. 
        }

        // �������� ����� ���� ����ġ ���Ű� ���
        for (int j = 0; j < numOfHiddenNodes; j++) {
            for (int k = 0; k < numOfOutputNodes; k++) {
                DWoutput[j * numOfOutputNodes + k] += Yhidden[i * numOfHiddenNodes + j] * Doutput[i * numOfOutputNodes + k];    
            }
        }

        // ������ ��Ÿ ���. Ȱ��ȭ �Լ� : �ñ׸��̵� �Լ�
        // Dhidden = Yhidden * (1 - Yhidden) * Doutput * tr(Woutput)
        for (int j = 0; j < numOfHiddenNodes; j++) {
            double dot = 0.0;
            for (int k = 0; k < numOfOutputNodes; k++) {
                dot += Doutput[i * numOfOutputNodes + k] * Woutput[j * numOfOutputNodes + k];
            }
            Dhidden[i * numOfHiddenNodes + j] = dot * Yhidden[i * numOfHiddenNodes + j] * (1 - Yhidden[i * numOfHiddenNodes + j]);  // ��Ÿ ���� ���̾�� ���Ű��� ����           
            DBhidden[j] += Dhidden[i * numOfHiddenNodes + j];       // race condition
        }

        // �Է����� ������ ���� ����ġ ���Ű� ���
        for (int j = 0; j < numOfXinputs; j++) {
            for (int k = 0; k < numOfHiddenNodes; k++) {
                DWhidden[j * numOfHiddenNodes + k] += Xinput[i * numOfXinputs + j] * Dhidden[i * numOfHiddenNodes + k];     
            }
        }
    }

    free(Thidden);
    free(Toutput);

    // �Է���-������ ����ġ ����
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] -= learningRate * DWhidden[i * numOfHiddenNodes + j];
        }
    }

    // ������ ���̾ �� ������-����� ����ġ ����
    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] -= learningRate * DBhidden[i];
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] -= learningRate * DWoutput[i * numOfOutputNodes + j];
        }
    }

    // ����� ���̾ ����
    for (int i = 0; i < numOfOutputNodes; i++) {
        Boutput[i] -= learningRate * DBoutput[i];
    }

    totalCost = totalCost / batchSize;
    printf("���� ��� = %lf\n", totalCost);

}

void trainingParallel(void) {

    // Whidden ��� ���ٽ� ĳ�� Ż���� �����ϱ� ���� ��ġ��ķ� ��ȯ
    double* Thidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Thidden[j * numOfXinputs + i] = Whidden[i * numOfHiddenNodes + j];
        }
    }

    // Woutput ��� ���ٽ� ĳ�� Ż���� �����ϱ� ���� ��ġ��ķ� ��ȯ
    double* Toutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    for (int i = 0; i < numOfHiddenNodes; i++) {
        for (int j = 0; j < numOfOutputNodes; j++) {
            Toutput[j * numOfHiddenNodes + i] = Woutput[i * numOfOutputNodes + j];
        }
    }

    double totalCost = 0.0;
 
    // ���� ó���� ���۵Ǵ� ��
#pragma omp parallel num_threads(4)
    {
        // ������ ���� ���� ����
        double localCost = 0.0;
        double * localDBoutput = (double *)calloc(numOfOutputNodes, sizeof(double));
        double * localDWoutput = (double *)calloc(numOfHiddenNodes * numOfOutputNodes, sizeof(double));
        double * localDBhidden = (double *)calloc(numOfHiddenNodes, sizeof(double));
        double * localDWhidden = (double *)calloc(numOfXinputs * numOfHiddenNodes, sizeof(double));
        int i = 0;

#pragma omp for
        for (i = 0; i < batchSize; i++) {
            // ������ ��� ���. Ȱ��ȭ �Լ� : �ñ׸��̵� �Լ�
            for (int j = 0; j < numOfHiddenNodes; j++) {
                double dot = 0.0;
                for (int k = 0; k < numOfXinputs; k++) {
                    dot += Xinput[i * numOfXinputs + k] * Thidden[j * numOfXinputs + k];
                }
                Yhidden[i * numOfHiddenNodes + j] = 1.0 / (1.0 + exp(-1.0 * (dot + Bhidden[j])));       // �ñ׸��̵� �Լ��� ���
            }

            // ����� ��� ���. Ȱ��ȭ �Լ� : ����Ʈ�ƽ� �Լ�
            double total = 0.0;
            double max = DBL_MIN;

            for (int j = 0; j < numOfOutputNodes; j++) {
                double dot = 0.0;
                for (int k = 0; k < numOfHiddenNodes; k++) {
                    dot += Yhidden[i * numOfHiddenNodes + k] * Toutput[j * numOfHiddenNodes + k];
                }
                Youtput[i * numOfOutputNodes + j] = dot + Boutput[j];
                if (Youtput[i * numOfOutputNodes + j] > max)
                    max = Youtput[i * numOfOutputNodes + j];                                            // �ִ밪 ã��  
            }

            for (int j = 0; j < numOfOutputNodes; j++) {
                total += exp(Youtput[i * numOfOutputNodes + j] - max);                                  // ����Ʈ�ƽ� �Լ��� �и� ���
            }

            for (int j = 0; j < numOfOutputNodes; j++) {
                Youtput[i * numOfOutputNodes + j] = exp(Youtput[i * numOfOutputNodes + j] - max) / total;     // ����Ʈ�ƽ� �Լ��� ���
            }

            // �������
            double errorCost = 0.0;
            for (int j = 0; j < numOfOutputNodes; j++) {
                if (Tinput[i] == j) {
                    errorCost += -1 * log(Youtput[i * numOfOutputNodes + j] + 0.0000001);           // ���� ��Ʈ���� ���
                }
            }
            localCost += errorCost;         // race condition

            // ����� ��Ÿ ���. Ȱ��ȭ �Լ� : ����Ʈ�ƽ� �Լ�
            // ��������� ����Ʈ�ƽ� �Լ��� ���� ��Ʈ���� ��� �Լ��� ����� ���
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
                localDBoutput[j] += Doutput[i * numOfOutputNodes + j];       // ��Ÿ ���� ���̾�� ���Ű��� ����. race condition
            }

            // �������� ����� ���� ����ġ ���Ű� ���
            for (int j = 0; j < numOfHiddenNodes; j++) {
                for (int k = 0; k < numOfOutputNodes; k++) {
                    localDWoutput[j * numOfOutputNodes + k] += Yhidden[i * numOfHiddenNodes + j] * Doutput[i * numOfOutputNodes + k];    // race condition
                }
            }

            // ������ ��Ÿ ���. Ȱ��ȭ �Լ� : �ñ׸��̵� �Լ�
            // Dhidden = Yhidden * (1 - Yhidden) * Doutput * tr(Woutput)
            for (int j = 0; j < numOfHiddenNodes; j++) {
                double dot = 0.0;
                for (int k = 0; k < numOfOutputNodes; k++) {
                    dot += Doutput[i * numOfOutputNodes + k] * Woutput[j * numOfOutputNodes + k];
                }
                Dhidden[i * numOfHiddenNodes + j] = dot * Yhidden[i * numOfHiddenNodes + j] * (1 - Yhidden[i * numOfHiddenNodes + j]);  // ��Ÿ ���� ���̾�� ���Ű��� ����           
                localDBhidden[j] += Dhidden[i * numOfHiddenNodes + j];       // race condition
            }

            // �Է����� ������ ���� ����ġ ���Ű� ���
            for (int j = 0; j < numOfXinputs; j++) {
                for (int k = 0; k < numOfHiddenNodes; k++) {
                    localDWhidden[j * numOfHiddenNodes + k] += Xinput[i * numOfXinputs + j] * Dhidden[i * numOfHiddenNodes + k];     // race condition
                }
            }
        }   // for �� ����
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

    // �Է���-������ ����ġ ����
    for (int i = 0; i < numOfXinputs; i++) {
        for (int j = 0; j < numOfHiddenNodes; j++) {
            Whidden[i * numOfHiddenNodes + j] -= learningRate * DWhidden[i * numOfHiddenNodes + j];
        }
    }

    // ������ ���̾ �� ������-����� ����ġ ����
    for (int i = 0; i < numOfHiddenNodes; i++) {
        Bhidden[i] -= learningRate * DBhidden[i];
        for (int j = 0; j < numOfOutputNodes; j++) {
            Woutput[i * numOfOutputNodes + j] -= learningRate * DWoutput[i * numOfOutputNodes + j];
        }
    }

    // ����� ���̾ ����
    for (int i = 0; i < numOfOutputNodes; i++) {
        Boutput[i] -= learningRate * DBoutput[i];
    }

    totalCost = totalCost / batchSize;
    printf("���� ��� = %lf\n", totalCost);

}

void calculateXWB(double* Y, double* X, double* W, double* B, int m, int n, int o, int type) {
    // Y = function(XW + B)
    // X : m * n ���
    // W : n * o ���
    // B : 1 * o ���
    // Y : m * o ���
    // function type : SIGMOID - sigmoid �Լ�, SOFTMAX - softmax �Լ� ����
    // W ��� ���ٽ� ĳ�� Ż���� �����ϱ� ���� ��ġ��ķ� ��ȯ
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
                Y[i * o + j] = 1.0 / (1.0 + exp(-1.0 * (dot + B[j])));  // �ñ׸��̵� �Լ��� ���
            }
            else {
                Y[i * o + j] = dot + B[j];
                total += exp(Y[i * o + j]);                             // ����Ʈ�ƽ� �Լ��� �и� ���
            }    
        }
        if (type == SOFTMAX) {
            for (int j = 0; j < o; j++) {
                Y[i * o + j] = exp(Y[i * o + j]) / total;               // ����Ʈ�ƽ� �Լ��� ���
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
        printf("Xdata �޸� �Ҵ� ����");
        exit(1);
    }

    Tdata = (int *)malloc(sizeof(int) * numOfCase);
    if (Tdata == NULL) {
        printf("Tdata �޸� �Ҵ� ����");
        exit(1);
    }

    Xinput = (double *)malloc(sizeof(double) * batchSize * numOfXinputs);
    if (Xinput == NULL) {
        printf("Xdata �޸� �Ҵ� ����");
        exit(1);
    }

    Tinput = (int *)malloc(sizeof(int) * batchSize);
    if (Tinput == NULL) {
        printf("Tinput �޸� �Ҵ� ����");
        exit(1);
    }

    Yhidden = (double *)malloc(sizeof(double) * batchSize * numOfHiddenNodes);
    if (Yhidden == NULL) {
        printf("Yhidden �޸� �Ҵ� ����");
        exit(1);
    }

    Youtput = (double *)malloc(sizeof(double) * batchSize * numOfOutputNodes);
    if (Youtput == NULL) {
        printf("Youtput �޸� �Ҵ� ����");
        exit(1);
    }

    Whidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    if (Whidden == NULL) {
        printf("Whidden �޸� �Ҵ� ����");
        exit(1);
    }

    Bhidden = (double *)malloc(sizeof(double) * numOfHiddenNodes);
    if (Bhidden == NULL) {
        printf("Bhidden �޸� �Ҵ� ����");
        exit(1);
    }

    Woutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    if (Woutput == NULL) {
        printf("Woutput �޸� �Ҵ� ����");
        exit(1);
    }

    Boutput = (double *)malloc(sizeof(double) * numOfOutputNodes);
    if (Boutput == NULL) {
        printf("Boutput �޸� �Ҵ� ����");
        exit(1);
    }

    DWhidden = (double *)malloc(sizeof(double) * numOfXinputs * numOfHiddenNodes);
    if (DWhidden == NULL) {
        printf("DWhidden �޸� �Ҵ� ����");
        exit(1);
    }

    DBhidden = (double *)malloc(sizeof(double) * numOfHiddenNodes);
    if (DBhidden == NULL) {
        printf("DBhidden �޸� �Ҵ� ����");
        exit(1);
    }

    DWoutput = (double *)malloc(sizeof(double) * numOfHiddenNodes * numOfOutputNodes);
    if (DWoutput == NULL) {
        printf("DWoutput �޸� �Ҵ� ����");
        exit(1);
    }

    DBoutput = (double *)malloc(sizeof(double) * numOfOutputNodes);
    if (DBoutput == NULL) {
        printf("DBoutput �޸� �Ҵ� ����");
        exit(1);
    }

    Dhidden = (double *)malloc(sizeof(double) * batchSize * numOfHiddenNodes);
    if (Dhidden == NULL) {
        printf("Dhidden �޸� �Ҵ� ����");
        exit(1);
    }

    Doutput = (double *)malloc(sizeof(double) * batchSize * numOfOutputNodes);
    if (Doutput == NULL) {
        printf("Doutput �޸� �Ҵ� ����");
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

    struct zip_t *zip = zip_open(zipfile, 0, 'r');              // ���� ���� �б�
    zip_entry_open(zip, fileName);                              // ������ ���� ���� ��, �޸𸮷� �ε�
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
        v1 = 2 * ((double)rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 ������ ��
        v2 = 2 * ((double)rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 ������ ��
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = sqrt((-2 * log(s)) / s);

    temp = v1 * s;
    temp = (stdev * temp) + average;
    
    return temp;
}