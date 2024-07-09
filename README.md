# kmeans

k-means clustering algorithm implementation in Go.

## Usage
Only has two functions, `Partition` and `PartitionWithCentroids`.

Use `PartitionWithCentroids` if you want to partition observations with your initial centroids. This can be used if you want to calculate initial centroids with any custom logic.

Use `Partition` if you don't care about the creation of initial centroids and they will be randomly genarated based on observations.
