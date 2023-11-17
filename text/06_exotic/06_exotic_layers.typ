= Skywarp: An attention-based model for the detection of Gravitational-Wave Compact Binary Coalescences <skywarp-sec>

== Attention! <sec-attention>

The global information provided by an individual element within a sequence is often greater than the local information contained within the isolated datum. This extra information is stored contextually within the relationship between the given element and the other elements in the sequence; both within the information stored locally by the other elements, and by the relative and absolute positions of the other elements.

The set of possible combinations of elements is large, even within relatively small sequences. Therefore, in order to regularise a machine learning model to extract contextual information efficiently, a method must be implemented to determine which elements contribute the most contextual information to a given datum. This method is attention. Attention determines which elements in the sequence contribute highly to the global information of a given element. Once attention has been determined, global contextual information can be embedded within each element’s local information. Ideally, this process makes the output elements, now with contextual information embedded locally, easier for other machine-learning methods to interpret. 

A transformer model is a machine learning algorithm which implements this method to localise global information using attention. The output to a transformer block has the same dimensionality as the block’s input, as it retains the same number of elements. Ideally, each element has been transformed to contain a proportion of the global information stored within the input sequence.

The models we describe in this section are novel in that they utilise attention mechanisms (Bahdanau et al. 2014,  Luong et al. 2015), a type of differentiable memory in which a global context vector is learnt over an input sequence $x_{i}$. 

The aim of the attention mechanism is to embed global context locally, in order to do this a comparison must be made between each element of the sequence and (in the case of self-attention) each other element of the same sequence. It is trivial to see that not every element in every sequence will be equally contextual to each other, and that the contextual dependence will depend on the information being extracted.

This corresponds to learning the following relation:

$ y_{j} = alpha^{i}_{j}x_{i} $



\begin{equation}
    
Where $alpha^{i}_{{i}j} in Re$ are the scalar attention scores measuring the relative correlation between $x_{i}$ and $x_{j}$. In this way, one learns intra-sequence relations; long-term dependencies are captured because the entire input sequence is used to compute a single element of the output sequence. 

In order to calculate these attention scores, we generate three vectors, namely a query $q_{i}$, key $k_{i}$, and value $v_{i}$ vector, for each sequence element $x_{i}$, forming three matrices for the sequence as a whole: $Q$, $K$, and $V$. These matrices are created by applying projection matrices: $W_{Q}$, $W_{K}$ and $W_{V}$ to the input sequence, the values of these weights matrices are learned via backpropagation during the model training process.

The query, key, and value matrices are used to calculate the attention scores for each element. The query value for each sequence element is matched against the key value of each other element, the alignment of the key and query determines a weighting for the value vector, a distilled representation of the information contained within that element. The weighted value vectors are then summed to produce the new, contextually embedded, sequence.

The two most commonly used attention functions are dot-product (Luong et al. 2015) and additive attention (Bahdanau et al. 2014), our models utilise the former and so we restrict our discussion to the work of (Luong et al.
2015) and extensions. In either case, the function $alpha$ maps a set of query $q$, key $k$ and value $v$ vectors to a weighted sum of the values. 

$ alpha(q, K, V) = sum_{i} a(q, k_i)v_i $
  
Where $a(., .)$ is called the alignment function and measures the similarity between the queries and keys. In the case of dot-product attention proposed by (Luong et al. 2015) :

$ a(q, k) = sigma(q^{T}k) $
  
Where $sigma$ is the Softmax function (). This calculation is performed on each element of the sequence to produce a new sequence of equal length, hopefully with some contextual context embedded. Generalising the attention function we get:

$ alpha(Q, K, V) = sigma(Q K^{T})V $
  

Where again $sigma$ is the Softmax function.

== Transformers

Since their introduction, attention mechanisms have been utilised in a number of different neural network architectures, including transformers and stable diffusion models. Transformers were first proposed by (Vaswani et al. 2017) to solve natural-language processing tasks, showing significant improvement over previous recurrent and convolutional architectures. For these reasons, we decided to investigate a fully attention-based model, inspired by a Transformer encoder.

The transformer model uses the attention mechanism described earlier in section @sec-attention within discrete blocks called multi-attention heads. Multi-attention heads have N multiples of the weights matrices used to generate query, key, and value matrices from input sequences. These multiple heads can be thought of in an analogous manner to different convolutional filters inside a CNN layer; each head can focus on extracting a different type of contextual information from the sequence. This is necessary as the useful contextual information embedded within a sequence can be more complex than it is possible to extract with a single attention head. The output sequences of all N heads are merged after the block to ensure that the output and input sizes are the same.

Often, as is the case for our models, the multi-attention heads are followed by normalisation and one or more dense layers. These blocks of layers can then be stacked to form components of a transformer.

%embedding
%positional encoding,

== Literature

Chatterjee et al offer perhaps the most relevant work, they utilize Long Short Term Memory (LSTM) networks, a form of Recurrent Neural Network, for both the problems of signal detection and reconstruction. Recurrent neural networks have an internal state determined by previous inferences, and thus they have the ability to retain some information about all previous data. In many ways, RNNs were the predecessor to Transformer models, largely because they are able, in some way, to make inferences from global information, rather than being limited by the receptive fields of convoloutional filters.

They have also been some attempts to detect other proposed signal types, known as gravitational-wave bursts, with machine learning methods, most prominently core-collapse supernovae. Although there has yet to be a real detection of any such signals, these types of signal are in many ways a more interesting candidate for machine learning methods, due to the much higher degree of uncertainty in waveform modeling. The physics behind burst signals are often much less understood, as well as possessing a much larger number of degrees of freedom. There have also been a few papers which have attempted to use machine learning methods as a coherence detection technique, therefore eschewing any requirement for accurate signals. Such detection problems could also benefit from the application of transformer models, and will be an area of future inquiry. 

As demonstrated, there has been a considerable investigation into the use of machine learning techniques for gravitational wave detection. However there has not been significant investigation into the use of transformers for such a purpose, with only this paper by Zhao et al known at this time, which focuses on the problem of space based detection. https://arxiv.org/pdf/2207.07414.pdf