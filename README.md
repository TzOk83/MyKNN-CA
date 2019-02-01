# MyKNN-CA
kNN of independent features classification algorithm

This is a C# implementation of my kNN of Independent Features Classification Algorithm.
This is a kNN classifier, which processes each of the original entity feature separately. This allows it to operate on non-fully defined entities - both for reference (learning) and query (classification). I have found it to achieve good classification results on multidimensional data (6+ features) with a significant amount of randomly placed missing (undefined) values (up to 35%). It doesn't require any missing value pre-processing, like filtering, or substituting.
