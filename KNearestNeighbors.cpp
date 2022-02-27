#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>

#include "KNearestNeighbors.h"

namespace
{
#define VOTE_SIZE 30
#define ENGINE_RANDOM_SEED std::chrono::system_clock::now().time_since_epoch().count()

  bool CompareDistAndSample(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
  {
      return lhs.first > rhs.second;
  }
}

namespace ArtificialIntelligence
{
namespace MachineLearning
{
namespace Supervised
{

namespace KNearestNeighbors
{

std::vector<std::vector<float>> NormalizeData(const std::vector<std::vector<float>>& data)
{
  const std::size_t size2D = data.size();
  const std::size_t size1D = data[0].size();
  std::vector<std::vector<float>> normalizedData = data;
  std::vector<float> max(size1D, std::numeric_limits<float>::min());
  std::vector<float> min(size1D, std::numeric_limits<float>::max());

  for (int i = 0; i < size2D; ++i)
  {
    for (int j = 0; j < size1D; ++j)
    {
      if (data[i][j] > max[j])
      {
        max[j] = data[i][j];
      }
      if (data[i][j] < min[j])
      {
        min[j] = data[i][j];
      }
    }
  }

  for (int i = 0; i < size2D; ++i)
  {
    for (int j = 0; j < size1D - 1; ++j)
    {
      normalizedData[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
    }
  }

  return std::move(normalizedData);
}

std::array<std::vector<std::vector<float>>, 2> SplitTrainTest(const std::vector<std::vector<float>>& normalizedData, const float percentTrain)
{
  const std::size_t size2D = normalizedData.size();
  const std::size_t size1D = normalizedData[0].size();
  std::vector<std::vector<float>> dataCopy = normalizedData;

  std::default_random_engine randomEngine(ENGINE_RANDOM_SEED);
  std::shuffle(std::begin(dataCopy), std::end(dataCopy), randomEngine);
  std::vector<std::vector<float>> train, test;
  std::size_t mid = static_cast<int>(static_cast<float>(size2D) * percentTrain);
  for (std::size_t i = 0; i < size2D; ++i)
  {
    if (i < mid)
    {
      train.push_back(dataCopy[i]);
    }
    else
    {
      test.push_back(dataCopy[i]);
    }
  }

  return std::array<std::vector<std::vector<float>>, 2>{ train, test };
}

std::vector<std::vector<std::vector<float>>> SplitCrossValidation(const std::vector<std::vector<float>>& normalizedData, const int kFold)
{
  const std::size_t size2D = normalizedData.size();
  const std::size_t size1D = normalizedData[0].size();
  std::vector<std::vector<float>> dataCopy = normalizedData;

  std::default_random_engine randomEngine(ENGINE_RANDOM_SEED);
  std::shuffle(std::begin(dataCopy), std::end(dataCopy), randomEngine);
  std::vector<std::vector<std::vector<float>>> split(kFold, std::vector<std::vector<float>>());

  for (int i = 0; i < size2D; ++i)
  {
    split[i % kFold].push_back(dataCopy[i]);
  }
  return split;
}

float Classify(const std::vector<std::vector<float>>& train, const std::vector<std::vector<float>>& test, const int nearestNeighborsCount)
{
  int numCorrect = 0;
  const std::size_t size2DTest = test.size();
  const std::size_t size1D = test[0].size(); // Same as train.
  const std::size_t size2DTrain = train.size();

  for (std::size_t i = 0; i < size2DTest; ++i)
  {
    std::vector<std::pair<float, int>> distAndSample;
    for (std::size_t j = 0; j < size2DTrain; ++j)
    {
      float distSqtPerSample = 0.f;
      for (std::size_t k = 0; k < size1D; ++k)
      {
        float a = test[i][k];
        float b = train[j][k];
        distSqtPerSample += (a - b) * (a - b);
      }
      distAndSample.push_back({ std::sqrt(distSqtPerSample), j });
    }
    std::sort(distAndSample.begin(), distAndSample.end(), CompareDistAndSample);
    float votes[VOTE_SIZE];
    std::fill(votes, votes + VOTE_SIZE, 0.f);

    // K-nearest neighbors. [0...K-1].
    for (int j = 0; j < nearestNeighborsCount; ++j)
    {
      int sampleIndex = distAndSample[j].second;
      int sampleLabel = static_cast<int>(train[sampleIndex][8]);
      votes[sampleLabel] += 1;
    }

    int voteLabelResult = 0, voteCount = 0;
    for (int j = 0; j < VOTE_SIZE; ++j)
    {
      if (votes[j] > voteCount)
      {
        voteLabelResult = j;
        voteCount = votes[j];
      }
    }
    int testLabelResult = static_cast<int>(test[i][8]);
    if (voteLabelResult == testLabelResult)
    {
      ++numCorrect;
    }
  }
  return static_cast<float>(numCorrect) / static_cast<float>(size2DTest);
}

} // namespace KNearestNeighbors

} // namespace SupervisedLearning
} // namespace MachineLearning
} // namespace ArtificialIntelligence
