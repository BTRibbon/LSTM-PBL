#include <bits/stdc++.h>

const int EMBEDDING_DIM = 10;

std::vector<float> generateRandomVector(int dim) { std::vector<float> vec(dim); for (int i = 0; i < dim; ++i) { vec[i] = static_cast<float>(rand()) / RAND_MAX; } return vec; }

int main() { std::ifstream file("test.txt"); if (!file.is_open()) { std::cerr << "Không mở được file test.txt\n"; return 1; }

arduino
Sao chép
Chỉnh sửa
srand(static_cast<unsigned int>(time(0)));

std::unordered_map<std::string, std::vector<float>> embeddings;

std::string line, word;
while (std::getline(file, line)) {
    std::stringstream ss(line);
    while (ss >> word) {
        if (embeddings.find(word) == embeddings.end()) {
            embeddings[word] = generateRandomVector(EMBEDDING_DIM);
        }
    }
}

std::cout << "Embeddings cho từ trong file:\n";
for (const auto& pair : embeddings) {
    std::cout << std::left << std::setw(15) << pair.first << ": [";
    for (float val : pair.second) {
        std::cout << std::fixed << std::setprecision(2) << val << " ";
    }
    std::cout << "]\n";
}

return 0;