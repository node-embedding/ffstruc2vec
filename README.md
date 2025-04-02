# ffstruc2vec

This repository provides a reference implementation of *ffstruc2vec* as described in the paper:<br>
> ffstruc2vec: Flat, Flexible and Scalable Learning of Node Representations from Structural Identities.<br>
> Mario Heidrich, Prof. Dr. Jeffrey Heidemann, Prof. Dr. Rüdiger Buchkremer, Prof. Dr. Gonzalo Wandosell Fernández de Bobadilla.<br>

The *ffstruc2vec* algorithm learns continuous representations for nodes in any graph. *ffstruc2vec* captures structural equivalence between nodes.  

Before executing ffstruc2vec, it is necessary to install the packages from the requirements.txt file.

### Basic Usage

#### Example
To run *ffstruc2vec* on Mirrored Zachary's karate club network with Default parameters, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/karate-mirrored.edgelist --output emb/karate-mirrored.emb``


#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int
		

#### Output

ffstruc2vec generates two output files.
The first file contains the learned node representation vectors and consists of *n+1* lines for a graph with *n* nodes.

The first line specifies the shape of the embedding and follows this format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *ffstruc2vec*.

The second output file contains a PCA visualization of the generated node representation vectors.

#### Flexibility

For example, you can set `--method 0` to enable Hyperopt-based tuning for your downstream task. Provide your labels via the `--path_labels` parameter using a format like the one in `graph/labels-brazil-airports.txt`.

This version of *ffstruc2vec* also allows you to manually assign weights to different *k*-hop neighborhoods and select the active graph feature (e.g., node degree, PageRank, etc.) via the `--active_feature` parameter.

You can view all available configuration options with: ```
python src/main.py --help
```

### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <node_embedding@gmail.com>.

*Note:* This is only a reference implementation of the framework *ffstruc2vec*.


<details>
<summary><strong>Full List of Command Line Options</strong></summary>

optional arguments:
  -h, --help            show this help message and exit
  --input [INPUT]       Input graph path
  --output [OUTPUT]     Embeddings path
  --dimensions DIMENSIONS
                        Number of dimensions. Default is 128.
  --walk-length WALK_LENGTH
                        Length of walk per source. Default is 80.
  --num-walks NUM_WALKS
                        Number of walks per source. Default is 10.
  --window-size WINDOW_SIZE
                        Context size for optimization. Default is 10.
  --until_layer UNTIL_LAYER
                        Calculation until the layer. Default is 3.
  --iter ITER           Number of epochs in SGD
  --workers WORKERS     Number of parallel workers. Default is 8.
  --weighted            Boolean specifying (un)weighted. Default is
                        unweighted.
  --unweighted
  --directed            Graph is (un)directed. Default is undirected.
  --undirected
  --OPT1 OPT1           optimization 1
  --OPT2 OPT2           optimization 2
  --OPT3 OPT3           optimization 3
  --weight_layer_0 WEIGHT_LAYER_0
                        Weight of layer 0 in case of the usage of the
                        flattened graph. Default is 1.
  --weight_layer_1 WEIGHT_LAYER_1
                        Weight of layer 1 in case of the usage of the
                        flattened graph. Default is 1.
  --weight_layer_2 WEIGHT_LAYER_2
                        Weight of layer 2 in case of the usage of the
                        flattened graph. Default is 1.
  --weight_layer_3 WEIGHT_LAYER_3
                        Weight of layer 3 in case of the usage of the
                        flattened graph. Default is 1.
  --weight_layer_4 WEIGHT_LAYER_4
                        Weight of layer 4 in case of the usage of the
                        flattened graph. Default is 1.
  --weight_layer_5 WEIGHT_LAYER_5
                        Weight of layer 5 in case of the usage of the
                        flattened graph. Default is 1.
  --weight_transformer WEIGHT_TRANSFORMER
                        Weight transformer for weight calculation out of
                        distances. Default is Euler's number e.
  --cost_function COST_FUNCTION
                        Cost function to be used to evaluate structural
                        similarity of nodes (0: DTW, 1: mean & variance, 2:
                        median & variance). Default is 1.
  --cost_calc COST_CALC
                        Cost calculatin to be used to evaluate structural
                        similarity of nodes (0: max(a, b) / min(a, b), 1:
                        max(a, b) - min(a, b) Default is 0.
  --mean_factor MEAN_FACTOR
                        Weighting factor for mean and var if mean&var distance
                        is used. Default is 0,5.
  --med_factor MED_FACTOR
                        Weighting factor for median and var if med&var
                        distance is used. Default is 0,5.
  --probability_distribution PROBABILITY_DISTRIBUTION
                        Probability distribution for choosing next node during
                        Deep Walk (0: linear distribution, 1: softmax).
                        Default is 0.
  --softmax_factor SOFTMAX_FACTOR
                        Applied factor in case of the usage of the flattened
                        graph with softmax probability distribution. Default
                        is 1.
  --path_labels PATH_LABELS
                        Path to labels for the application of an
                        classification algorithm. Default is "".
  --active_feature ACTIVE_FEATURE
                        Feature of nodes to be used to calculate the distances
                        between the nodes. (0: node degree, 1: pagerank, 2:
                        closeness_centrality, 3: betweenness_centrality, 4:
                        eigenvector_centrality, 5: core_number, 6:
                        clustering). Default is 0.
  --pagerank_scale PAGERANK_SCALE
                        Applied factor in case of the usage of the pagerank.
                        Default is 1.
  --method METHOD       Method to be applied (0: Hyperopt Classification, 1:
                        Simple Classification, 2: Hyperopt Clustering, 3: Only
                        embedding) Default is 3.


---

## License

This project is licensed under the **Apache License 2.0**. You are free to use, modify, and distribute the code, provided you adhere to the terms of the license. See the [LICENSE](./LICENSE) file for details.

This project also includes components adapted from the original **struc2vec** implementation by Leonardo Filipe Rodrigues Ribeiro, which is licensed under the **MIT License**. These components are located in selected parts of the codebase and documented accordingly.

For details, see the included [LICENSE_STRUC2VEC](./LICENSE_STRUC2VEC) file.
