#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include "KNearestNeighbors.h"

using namespace ArtificialIntelligence::MachineLearning::Supervised;

std::vector<std::string> TokenizeString(const std::string& line, const char separator);
std::vector<std::vector<float>> LoadData(std::string const& fileName, const char separator);
std::ostream& operator<<(std::ostream& outputStream, const std::vector<float>& vector1DData);
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& lhs, const std::vector<std::vector<float>>& rhs);
void TestNormalizedData(const std::vector<std::vector<float>>& normalizedData);
void TestTrainTestKNearestNeighbors(std::function<float(const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&, const int)> learnFn
  , const std::vector<std::vector<float>>& normalizedData
  , const float percentTrain
  , const int nearestNeighborsCount);
void TestCrossValidationKNearestNeighbors(std::function<float(const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&, const int)> learnFn
  , const std::vector<std::vector<float>>& normalizedData
  , const int kFold
  , const int nearestNeighborsCount);

int main()
{
  std::vector<std::vector<float>> rawData = LoadData("abalone.data", ',');
  if (rawData.empty())
  {
    return -1;
  }

  std::vector<std::vector<float>> normalizedData = KNearestNeighbors::NormalizeData(rawData);
  //TestNormalizedData(normalizedData);
  /*for (auto& vector1DData : normalizedData)
  {
    std::cout << vector1DData << "\n";
  }*/

  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.7f, 1);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.6f, 1);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.5f, 1);
  TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.7f, 5);
  TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.6f, 5);
  TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.5f, 5);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.7f, 10);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.6f, 10);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.5f, 10);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.7f, 15);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.6f, 15);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.5f, 15);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.7f, 20);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.6f, 20);
  //TestTrainTestKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 0.5f, 20);

  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 5, 1);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 10, 1);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 15, 1);
  TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 5, 5);
  TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 10, 5);
  TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 15, 5);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 5, 10);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 10, 10);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 15, 10);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 5, 15);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 10, 15);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 15, 15);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 5, 20);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 10, 20);
  //TestCrossValidationKNearestNeighbors(KNearestNeighbors::Classify, normalizedData, 15, 20);
}

std::vector<std::string> TokenizeString(const std::string& line, const char separator)
{
  std::vector<std::string> results;
  std::stringstream check(line);
  std::string intermediate;

  while(getline(check, intermediate, separator))
  {
      results.push_back(intermediate);
  }
  return std::move(results);
}

std::vector<std::vector<float>> LoadData(const std::string& fileName, const char separator)
{
  std::string line;
  std::ifstream file(fileName);
  if (!file.is_open())
  {
    return std::vector<std::vector<float>>();
  }

  std::vector<std::vector<float>> results;
  int index = 0;
  while (getline(file, line))
  {
    results.push_back(std::vector<float>());
    std::vector<std::string> words = TokenizeString(line, separator);
    for (auto& word : words)
    {
      if (word == "M")
        word = "0.333";
      else if (word == "F")
        word = "0.666";
      else if (word == "I")
        word = "1.0";
      results[index].push_back(std::stof(word));
    }
    ++index;
  }
  file.close();
  return std::move(results);
}

std::ostream& operator<<(std::ostream& outputStream, const std::vector<float>& vector1DData)
{
  outputStream << "[ ";
  const std::size_t size1D = vector1DData.size();
  for (std::size_t i = 0; i < size1D; ++i)
  {
    outputStream << std::fixed << vector1DData[i];
    if (i < size1D - 1)
    {
      outputStream << ", ";
    }
  }
  outputStream << "]";
  return outputStream;
}

std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>>& lhs, const std::vector<std::vector<float>>& rhs)
{
  std::vector<std::vector<float>> result = lhs;
  for (auto const& right : rhs)
  {
    result.push_back(right);
  }
  return std::move(result);
}

void TestNormalizedData(const std::vector<std::vector<float>>& normalizedData)
{
  std::vector<float> sum = normalizedData[0];
  for (int i = 1; i < normalizedData.size(); ++i)
  {
    for (int j = 0; j < normalizedData[0].size(); ++j)
    {
      sum[j] += normalizedData[i][j];
    }
  }

  std::vector<float> mean = sum;
  for (int j = 0; j < mean.size(); ++j)
  {
    mean[j] /= static_cast<float>(normalizedData.size());
  }
  std::cout << "mean:" << mean << "\n";
  std::cout << "sum:" << sum << "\n";
}

void TestTrainTestKNearestNeighbors(std::function<float(const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&, const int)> learnFn
  , const std::vector<std::vector<float>>& normalizedData
  , const float percentTrain
  , const int nearestNeighborsCount)
{
  auto start = std::chrono::high_resolution_clock::now();
  std::array<std::vector<std::vector<float>>, 2> trainTest = KNearestNeighbors::SplitTrainTest(normalizedData, percentTrain);
  float accuracy = learnFn(trainTest[0], trainTest[1], nearestNeighborsCount);
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout
    << "Percent Train " << percentTrain * 100.f << "% Accuracy: " << accuracy
    << ", time: " << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) / 1000000.f << "s"
    << ", " << nearestNeighborsCount << "-nearest-neighbors"
    << "\n";
}

void TestCrossValidationKNearestNeighbors(std::function<float(const std::vector<std::vector<float>>&, const std::vector<std::vector<float>>&, const int)> learnFn
  , const std::vector<std::vector<float>>& normalizedData
  , const int kFold
  , const int nearestNeighborsCount)
{
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<std::vector<float>>> crossValidation = KNearestNeighbors::SplitCrossValidation(normalizedData, kFold);
  std::array<std::vector<std::vector<float>>, 2> trainTest{
    crossValidation[0] + crossValidation[1] + crossValidation[2] + crossValidation[3],
    crossValidation[4]
  };
  float accuracy = learnFn(trainTest[0], trainTest[1], nearestNeighborsCount);
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout
    << "Cross Validation " << kFold << "-Fold Accuracy: " << accuracy
    << ", time: " << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()) / 1000000.f << "s"
    << ", " << nearestNeighborsCount << "-nearest-neighbors"
    << "\n";
}
