#include <array>
#include <vector>

namespace ArtificialIntelligence
{
namespace MachineLearning
{
namespace Supervised
{

namespace KNearestNeighbors 
{
  // Param1 "data": the loaded dataset. [2D array]
  // Return: the normalized dataset. [2D array]
  std::vector<std::vector<float>> NormalizeData(const std::vector<std::vector<float>>& data);
  // Param1 "normalizedData": normalized dataset. [2D array]
  // Param2 "percentTrain": expected portion of dataset for training, the domain is (0..1).
  // Return: a list contain two elements, the first is train in type of 2D array, the second is test in type of 2D array.
  std::array<std::vector<std::vector<float>>, 2> SplitTrainTest(const std::vector<std::vector<float>>& normalizedData, const float percentTrain);
  // Param1 "normalizedData": normalized dataset. [2D array]
  // Param2 "kFold": k-fold. [int]
  // Return: a list contain k elements, each element is in type of 2D array.
  std::vector<std::vector<std::vector<float>>> SplitCrossValidation(const std::vector<std::vector<float>>& normalizedData, const int kFold);

  // Param1 "train": dataset for training. [2D array]
  // Param2 "test": dataset for testing. [2D array]
  // Param3 "nearestNeighborsCount": number of Nearest neighbors.
  // Return: Classification accuracy.
  float Classify(const std::vector<std::vector<float>>& train, const std::vector<std::vector<float>>& test, const int nearestNeighborsCount);

} // namespace KNearestNeighbors

} // namespace SupervisedLearning
} // namespace MachineLearning
} // namespace ArtificialIntelligence
