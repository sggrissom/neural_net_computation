#include <iostream>
#include <time.h>
#include "BackProp.h"
// NeuralNet.cpp : Defines the entry point for the console application.
//

#include "backprop.h"
#include "dataParser.h"

using std::cout;
using std::endl;

int predictData(double data[],
                int inputSize,
                int outputSize,
                int dataPoints,
                int testPoints);

int main(int argc, char* argv[])
{
    //0 - mushroom
    //1 - random test
    //2 - xor
    
    int mode = 0;
    
    if(argc == 2) {
        mode = atoi(argv[1]);
    }
    
    int inputSize;
    int outputSize = 1;
    int dataPoints;
    int testPoints = 8;
    
    double * dataToTest;
    
    double mushroomData[MAX_LINES * MAX_TOKENS_PER_LINE] = {};
    
    getTestData("/Users/sggrissom/Desktop/final_proj/mushroom.data",
                mushroomData);
    
    double data[]={
        1,2,2,2,2,0.5,
        3,4,2,2,1,0.5,
        5,6,2,1,2,0.5,
        7,8,1,2,2,0.5,
        9,10,1,1,2,0.1,
        11,12,2,1,1,0.1,
        13,14,1,2,1,0.1,
        15,16,1,1,1,1};
    
    // prepare XOR traing data
    
    double xorData[]={
        0,	0,	0,	0,
        0,	0,	1,	1,
        0,	1,	0,	1,
        0,	1,	1,	0,
        1,	0,	0,	1,
        1,	0,	1,	0,
        1,	1,	0,	0,
        1,	1,	1,	1 };
    
    switch (mode) {
        case 0: {
            inputSize = MAX_TOKENS_PER_LINE-1;
            dataPoints = MAX_LINES;
            dataToTest = mushroomData;
        } break;
        case 1: {
            inputSize = 5;
            dataPoints = 8;
            dataToTest = data;
        } break;
        case 2:
        default: {
            inputSize = 3;
            dataPoints = 8;
            dataToTest = xorData;
        }
    }
    
    predictData(dataToTest, inputSize, outputSize, dataPoints, testPoints);
}

int predictData(double data[],
                int inputSize,
                int outputSize,
                int dataPoints,
                int testPoints)
{
    
    // defining a net with 4 layers having 3,3,3, and 1 neuron respectively,
    // the first layer is input layer i.e. simply holder for the input parameters
    // and has to be the same size as the no of input parameters, in out example 3
    int numLayers = 4, lSz[4] = {inputSize,
                                inputSize-1,
                                inputSize-2,
                                outputSize};
    
    
    // Learing rate - beta
    // momentum - alpha
    // Threshhold - thresh (value of target mse, training stops once it is achieved)
    double beta = 0.3, alpha = 0.1, Thresh =  0.3;
    

    // maximum no of iterations during training
    long num_iter = 20000000;
    
    // Creating the net
    CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);
    
    long i;
    double mse, maxMse = 0;
    
    t1 = clock();
    
    cout<< endl <<  "Now training the network...." << endl;
    for (i=0; i<num_iter ; i++)
    {
        
        bp->bpgt(&data[(i%dataPoints)*(inputSize+outputSize)],
                 &data[(i%dataPoints)*(inputSize+outputSize) + inputSize]);
        
        mse = bp->mse(&data[(i%dataPoints)*(inputSize+outputSize) + inputSize]);
        if (mse>maxMse) maxMse = mse;
       
        if((i%dataPoints == dataPoints-1) && maxMse < Thresh) {
            cout << endl << "Network Trained. Threshold value achieved in " << i << " iterations." << endl;
            cout << "MSE:  " << bp->mse(&data[(i%dataPoints)*(inputSize+outputSize) + inputSize])
            <<  endl <<  endl;
            break;
        }
        if ( i%(num_iter/10) == 0 )
            cout<<  endl <<  "MSE:  " << bp->mse(&data[(i%dataPoints)*(inputSize+outputSize) + inputSize])
            << "... Training..." << endl;
        
    }
    
    if ( i == num_iter )
        cout << endl << i << " iterations completed..."
        << "MSE: " << maxMse << endl;
    
    t2 = clock();
    trainClockDiff = (float)t2 - (float)t1;
    
    cout<< "Now using the trained network to make predctions on test data...." << endl << endl;
    for ( i = 0 ; i < testPoints ; ++i )
    {
        bp->ffwd(&data[i*(inputSize+outputSize)]);
        for (int j=0; j < inputSize; ++j)
        {
            cout << data[i*(inputSize+outputSize)+j] << " ";
        }
        cout << "Ans:" << data[i*(inputSize+outputSize) + inputSize] <<  "  Guess:" << bp->Out(0) << endl;
    }
    
    int prediction, actual;
    float guess;
    int correct = 0, incorrect = 0;
    
    t1 = clock();
    for (int i = 0; i < dataPoints; ++i)
    {
        bp->ffwd(&data[i*(inputSize+outputSize)]);
        actual = data[i*(inputSize+outputSize) + inputSize];
        guess = bp->Out(0);
        prediction = (int)(guess + 0.5);
        if (prediction == actual)
        {
            correct++;
        } else {
            incorrect++;
        }
    }
    
    t2 = clock();
    predictClockDiff = (float)t2 - (float)t1;
     
    float accuracy = ((float)correct)/((float)(correct+incorrect));
    
    cout << "Correct = " << correct << endl;
    cout << "Incorrect = " << incorrect << endl;
    cout << "Accuracy = " << accuracy*100 << "%" << endl;
    cout << "Train Time: " << trainClockDiff/CLOCKS_PER_SEC << "s\n";
    cout << "Predict Time: " << predictClockDiff/CLOCKS_PER_SEC << "s\n";
    return 0;
}
