#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

struct Sample {
    vector<double> input;
    double output;
};

double activateSigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

vector<double> activateSigmoid(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = activateSigmoid(vec[i]);
    return res;
}

double activateTanh(double x) {
    return tanh(x);
}

vector<double> activateTanh(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = activateTanh(vec[i]);
    return res;
}

double dActivateSigmoid(double y) {
    return y * (1.0 - y);
}

vector<double> dActivateSigmoid(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = vec[i] * (1.0 - vec[i]);
    return res;
}

double dActivateTanh(double y) {
    return 1.0 - y * y;
}

vector<double> dActivateTanh(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = 1.0 - vec[i] * vec[i];
    return res;
}

vector<double> addVectors(const vector<double>& a, const vector<double>& b) {
    vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); i++)
        res[i] = a[i] + b[i];
    return res;
}

vector<double> multiplyVectors(const vector<double>& a, const vector<double>& b) {
    vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); i++)
        res[i] = a[i] * b[i];
    return res;
}

vector<double> scaleVector(const vector<double>& vec, double factor) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = vec[i] * factor;
    return res;
}

vector<double> matrixVectorProduct(const vector<vector<double>>& mat, const vector<double>& vec) {
    vector<double> res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); i++)
        for (size_t j = 0; j < vec.size(); j++)
            res[i] += mat[i][j] * vec[j];
    return res;
}

vector<double> concatVectors(const vector<double>& v1, const vector<double>& v2) {
    vector<double> res = v1;
    res.insert(res.end(), v2.begin(), v2.end());
    return res;
}

double clipValue(double val, double minVal, double maxVal) {
    return max(minVal, min(val, maxVal));
}

vector<double> clipVector(const vector<double>& vec, double minVal, double maxVal) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = clipValue(vec[i], minVal, maxVal);
    return res;
}

class MyLSTM {
    int inSize;
    int hidSize;
    vector<vector<double>> Wf, Wi, Wc, Wo;
    vector<double> bf, bi, bc, bo;
    
    // Lưu trữ các kết quả trung gian
    vector<double> lastInput, lastHidden, lastCell;
    vector<double> gateForget, gateInput, candidate, gateOutput;
    vector<double> cellState, hiddenState;
    
    mt19937 rng;
    normal_distribution<double> dist;
    
public:
    MyLSTM(int inputSize, int hiddenSize)
        : inSize(inputSize), hidSize(hiddenSize), rng(random_device{}()), dist(0.0, 0.1) {
        Wf = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        Wi = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        Wc = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        Wo = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        bf = vector<double>(hidSize, 0.0);
        bi = vector<double>(hidSize, 0.0);
        bc = vector<double>(hidSize, 0.0);
        bo = vector<double>(hidSize, 0.0);
        
        // Khởi tạo trọng số với giá trị nhỏ ngẫu nhiên
        for (int i = 0; i < hidSize; i++) {
            for (int j = 0; j < inSize + hidSize; j++) {
                Wf[i][j] = dist(rng);
                Wi[i][j] = dist(rng);
                Wc[i][j] = dist(rng);
                Wo[i][j] = dist(rng);
            }
            bf[i] = 1.0; // Thiết lập bias của forget gate
        }
    }
    
    // Hàm forward cho một bước thời gian
    pair<vector<double>, vector<double>> forward(const vector<double>& input, 
                                                   const vector<double>& prevHidden, 
                                                   const vector<double>& prevCell) {
        lastInput = input;
        lastHidden = prevHidden;
        lastCell = prevCell;
        vector<double> combined = concatVectors(input, prevHidden);
        
        gateForget = activateSigmoid(addVectors(matrixVectorProduct(Wf, combined), bf));
        gateInput  = activateSigmoid(addVectors(matrixVectorProduct(Wi, combined), bi));
        candidate  = activateTanh(addVectors(matrixVectorProduct(Wc, combined), bc));
        cellState  = addVectors(multiplyVectors(gateForget, prevCell),
                                 multiplyVectors(gateInput, candidate));
        gateOutput = activateSigmoid(addVectors(matrixVectorProduct(Wo, combined), bo));
        hiddenState = multiplyVectors(activateTanh(cellState), gateOutput);
        
        return {hiddenState, cellState};
    }
    
    // Hàm backward cho một bước thời gian
    // dHiddenNext: gradient của hidden state từ bước sau
    // dCellNext: gradient của cell state từ bước sau
    // lr: learning rate

    //this is insane tbh
    
    tuple<vector<double>, vector<double>, vector<double>> backward(const vector<double>& dHiddenNext, 
                                                                     const vector<double>& dCellNext, 
                                                                     double lr, 
                                                                     double clipVal = 5.0) {
        vector<double> dGateOutput = multiplyVectors(dHiddenNext, activateTanh(cellState));
        vector<double> dCellTotal = addVectors(dCellNext, 
                                    multiplyVectors(dHiddenNext, 
                                        multiplyVectors(gateOutput, dActivateTanh(activateTanh(cellState)))));
        
        vector<double> dGateInput  = multiplyVectors(dCellTotal, candidate);
        vector<double> dCandidate  = multiplyVectors(dCellTotal, gateInput);
        vector<double> dGateForget = multiplyVectors(dCellTotal, lastCell);
        
        // Áp dụng đạo hàm của các hàm kích hoạt
        dGateOutput = multiplyVectors(dGateOutput, dActivateSigmoid(gateOutput));
        dGateInput  = multiplyVectors(dGateInput, dActivateSigmoid(gateInput));
        dCandidate  = multiplyVectors(dCandidate, dActivateTanh(candidate));
        dGateForget = multiplyVectors(dGateForget, dActivateSigmoid(gateForget));
        
        // Cắt gradient cho an toàn
        dGateOutput = clipVector(dGateOutput, -clipVal, clipVal);
        dGateInput  = clipVector(dGateInput, -clipVal, clipVal);
        dCandidate  = clipVector(dCandidate, -clipVal, clipVal);
        dGateForget = clipVector(dGateForget, -clipVal, clipVal);
        
        vector<double> combined = concatVectors(lastInput, lastHidden);
        
        // Cập nhật trọng số cho mỗi cổng
        for (int i = 0; i < hidSize; i++) {
            for (int j = 0; j < inSize + hidSize; j++) {
                Wf[i][j] -= lr * dGateForget[i] * combined[j];
                Wi[i][j] -= lr * dGateInput[i]  * combined[j];
                Wc[i][j] -= lr * dCandidate[i]  * combined[j];
                Wo[i][j] -= lr * dGateOutput[i] * combined[j];
            }
            bf[i] -= lr * dGateForget[i];
            bi[i] -= lr * dGateInput[i];
            bc[i] -= lr * dCandidate[i];
            bo[i] -= lr * dGateOutput[i];
        }
        
        // Tính gradient cho hidden state và cell state của bước trước
        vector<double> dCombined(inSize + hidSize, 0.0);
        for (int i = 0; i < hidSize; i++) {
            for (int j = 0; j < inSize + hidSize; j++) {
                dCombined[j] += Wf[i][j] * dGateForget[i] +
                                Wi[i][j] * dGateInput[i] +
                                Wc[i][j] * dCandidate[i] +
                                Wo[i][j] * dGateOutput[i];
            }
        }
        vector<double> dPrevInput(dCombined.begin(), dCombined.begin() + inSize);
        vector<double> dPrevHidden(dCombined.begin() + inSize, dCombined.end());
        vector<double> dPrevCell = multiplyVectors(dCellTotal, gateForget);
        
        // trả về gradient của input, hidden state và cell state.
        // Gradient trọng số có thể được tính riêng nếu cần.
        return {dPrevInput, dPrevHidden, dPrevCell};
    }
};

class LSTMModel {
    int inputDim;
    int hiddenDim;
    int outputDim;
    MyLSTM lstmCell;
    vector<vector<double>> Wy;
    vector<double> by;
    double learningRate;
    int epochs;
    
public:
    LSTMModel(int inDim, int hidDim, int outDim, double lr = 0.01, int ep = 50)
        : inputDim(inDim), hiddenDim(hidDim), outputDim(outDim), lstmCell(inDim, hidDim),
          learningRate(lr), epochs(ep) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> d(0.0, 0.1);
        Wy = vector<vector<double>>(outputDim, vector<double>(hiddenDim));
        by = vector<double>(outputDim, 0.0);
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < hiddenDim; j++)
                Wy[i][j] = d(gen);
        }
    }
    
    double trainModel(const vector<Sample>& dataset) {
        double totalLoss = 0.0;
        for (int ep = 0; ep < epochs; ep++) {
            double epLoss = 0.0;
            for (const auto& s : dataset) {
                vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
                auto [hOut, cOut] = lstmCell.forward(s.input, h, c);
                
                // Lớp output
                vector<double> pred(outputDim, 0.0);
                vector<double> temp = matrixVectorProduct(Wy, hOut);
                for (int i = 0; i < outputDim; i++)
                    pred[i] = temp[i] + by[i];
                
                // Tính loss theo MSE
                double loss = 0.0;
                vector<double> dPred(outputDim, 0.0);
                for (int i = 0; i < outputDim; i++) {
                    double err = pred[i] - s.output;
                    loss += err * err;
                    dPred[i] = 2.0 * err;
                }
                loss /= outputDim;
                epLoss += loss;
                
                // Backprop lớp output
                vector<double> dHidden(hiddenDim, 0.0);
                for (int i = 0; i < outputDim; i++) {
                    for (int j = 0; j < hiddenDim; j++) {
                        Wy[i][j] -= learningRate * dPred[i] * hOut[j];
                        dHidden[j] += Wy[i][j] * dPred[i];
                    }
                    by[i] -= learningRate * dPred[i];
                }
                
                vector<double> dCell(hiddenDim, 0.0);
                lstmCell.backward(dHidden, dCell, learningRate);
            }
            epLoss /= dataset.size();
            totalLoss = epLoss;
            if ((ep + 1) % 10 == 0 || ep == 0)
                cout << "Epoch " << (ep + 1) << "/" << epochs << " - Loss: " << epLoss << endl;
        }
        return totalLoss;
    }
    
    double predictOutput(const vector<double>& inputData) {
        vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
        auto [hOut, cOut] = lstmCell.forward(inputData, h, c);
        vector<double> out = matrixVectorProduct(Wy, hOut);
        for (int i = 0; i < outputDim; i++)
            out[i] += by[i];
        return out[0];
    }
};

string decodeOutput(double val) {
    int category = static_cast<int>(round(val));
    switch (category) {
        case 1: return "Mưa";
        case 2: return "Nắng";
        case 3: return "Tuyết";
        case 4: return "Phùn";
        case 5: return "Sương mù";
        default: return "Không xác định";
    }
}

int main() {
    int inDim = 1;
    int hidDim = 32;
    int outDim = 1;
    double lr = 0.01;
    int numEpochs = 1000;
    
    vector<Sample> trainData;
    freopen("data.txt", "r", stdin);
    for (int i = 0; i < 1460; i++) {
        Sample s;
        vector<double> feat;
        for (int j = 0; j < 5; j++) {
            double temp;
            cin >> temp;
            feat.push_back(temp);
        }
        s.input = feat;
        cin >> s.output;
        trainData.push_back(s);
    }
    
    LSTMModel model(inDim, hidDim, outDim, lr, numEpochs);
    double finalLoss = model.trainModel(trainData);
    cout << "Training completed. Final Loss: " << finalLoss << endl;
    
    cout << "Testing prediction:" << endl;
    vector<double> testInput;
    for (int i = 0; i < 5; i++) {
        double val;
        cin >> val;
        testInput.push_back(val);
    }
    double actual;
    cin >> actual;
    double pred = model.predictOutput(testInput);
    cout << "Dự đoán cho ngày 31/12/2015: Actual: " << decodeOutput(actual)
         << ", Predicted: " << decodeOutput(pred) << endl;
    cout << "Hehe i know u'r reading :)))" << endl;
    
    return 0;
}
