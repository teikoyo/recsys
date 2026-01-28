# Pseudocode for analyse_pytorch.ipynb

## STEP 1: Environment & Utilities Setup

```
IMPORT PyTorch, FAISS, NumPy, Pandas, SciPy.sparse

FUNCTION set_seed(seed):
    set random.seed to seed
    set numpy.random.seed to seed
    set torch.manual_seed to seed
    set torch.cuda.manual_seed_all to seed

FUNCTION get_device():
    IF CUDA is available THEN:
        RETURN torch.device('cuda:0')
    ELSE:
        RETURN torch.device('cpu')

FUNCTION csr_to_torch_coo(sparse_matrix, device):
    convert sparse_matrix to COO format
    extract row_indices, col_indices, values from COO
    create PyTorch sparse tensor with (row_indices, col_indices, values)
    move tensor to device
    RETURN sparse tensor

FUNCTION torch_row_l2_normalize(sparse_tensor, epsilon):
    extract indices and values from sparse_tensor
    get row_indices from indices[0]

    // Compute squared values per row
    squared_values = values ** 2
    initialize row_norms_squared as zeros with size = number of rows

    FOR each position in squared_values:
        row_id = row_indices[position]
        add squared_values[position] to row_norms_squared[row_id]

    // Compute L2 norms
    row_norms = square_root(row_norms_squared + epsilon)

    // Normalize each value by its row's norm
    normalized_values = empty array of same size as values
    FOR each position in values:
        row_id = row_indices[position]
        normalized_values[position] = values[position] / row_norms[row_id]

    create new sparse tensor with normalized_values
    RETURN normalized sparse tensor

FUNCTION save_csr_as_triplets(sparse_matrix, filepath):
    convert sparse_matrix to COO format
    create DataFrame with columns: [row, col, data]
    save DataFrame to filepath as parquet

FUNCTION load_csr_from_triplets(filepath, shape):
    load DataFrame from filepath
    extract row, col, data from DataFrame
    create CSR matrix with (data, (row, col)) and given shape
    RETURN CSR matrix

// Initialize environment
CALL set_seed(2025)
device = CALL get_device()
set torch float32 matmul precision to 'high'
enable torch cudnn benchmark mode
```

---

## STEP 2: Data Preprocessing & Cleaning

```
// Load raw data
raw_data = read CSV file from RAW_CSV_PATH

// Select time column
time_column = NULL
FOR each column in raw_data.columns:
    IF column ends with '_dt' OR column contains 'time' THEN:
        time_column = column
        BREAK

// Remove duplicate IDs
raw_data = keep only first occurrence of each ID

// Merge text columns
text_columns = [title, description, content]
FOR each row in raw_data:
    text_pieces = empty list
    FOR each column in text_columns:
        IF row[column] is not null THEN:
            append row[column] to text_pieces

    row.text_all = join text_pieces with space separator

// Clean text
FOR each row in raw_data:
    text = row.text_all

    // Remove HTML tags
    text = replace pattern "<[^>]+>" with empty string in text

    // Remove URLs
    text = replace pattern "http[s]?://\S+" with empty string in text

    // Convert to lowercase
    text = convert text to lowercase

    // Strip whitespace
    text = remove leading and trailing whitespace from text

    row.text_all = text

// Parse tags
FOR each row in raw_data:
    tag_string = row.tags

    IF tag_string is null THEN:
        row.tag_str = empty string
        CONTINUE

    IF tag_string starts with '[' THEN:
        // JSON format
        tags = parse tag_string as JSON array
    ELSE:
        // Delimiter-separated format
        tags = split tag_string by characters [|, ;]
        FOR each tag in tags:
            tag = strip whitespace from tag
            tag = convert tag to lowercase

    row.tag_str = join tags with comma separator

// Create final document dataframe
doc_df = select columns [id, text_all, tag_str, created_time] from raw_data
doc_df.internal_idx = range from 0 to N-1

// Create index mapping
index_map = select columns [id, internal_idx] from doc_df

// Save to disk
save doc_df to 'doc_clean.parquet'
save index_map to 'index_map.parquet'
```

---

## STEP 3: Tag View Construction (D-T Matrix with 3 Weightings)

```
// Load cleaned documents
doc_df = load parquet from 'doc_clean.parquet'
N = number of documents in doc_df

FUNCTION build_tag_vocabulary(documents, min_document_frequency):
    initialize tag_counter as empty dictionary

    // Count document frequency for each tag
    FOR each document in documents:
        IF document.tag_str is not null THEN:
            tags = split document.tag_str by comma
            unique_tags = get unique tags from tags

            FOR each tag in unique_tags:
                tag = strip whitespace from tag
                IF tag is not empty THEN:
                    increment tag_counter[tag] by 1

    // Filter by minimum document frequency
    vocabulary = empty dictionary
    tag_index = 0
    FOR each (tag, count) in tag_counter:
        IF count >= min_document_frequency THEN:
            vocabulary[tag] = {
                'idx': tag_index,
                'df': count
            }
            increment tag_index

    RETURN vocabulary

FUNCTION build_DT_binary(documents, tag_vocabulary):
    rows = empty list
    cols = empty list

    // Build sparse matrix entries
    FOR doc_idx FROM 0 TO number of documents - 1:
        tag_str = documents[doc_idx].tag_str

        IF tag_str is not null AND not empty THEN:
            tags = split tag_str by comma
            unique_tags = get unique tags from tags

            FOR each tag in unique_tags:
                tag = strip whitespace from tag
                IF tag is in tag_vocabulary THEN:
                    tag_idx = tag_vocabulary[tag]['idx']
                    append doc_idx to rows
                    append tag_idx to cols

    // Create sparse matrix with all values = 1.0
    values = array of ones with length = length of rows
    DT_binary = create CSR matrix with (values, (rows, cols))
    set matrix shape to (N, V) where V = size of tag_vocabulary

    RETURN DT_binary

FUNCTION compute_tfidf_weights_gpu(DT_binary, tag_vocabulary, smoothing):
    convert DT_binary to COO format
    extract row_indices, col_indices from COO

    // Move to GPU
    col_indices_gpu = move col_indices to GPU device

    // Build IDF vector
    V = size of tag_vocabulary
    idf_vector = create zero vector of size V on GPU

    FOR each (tag, info) in tag_vocabulary:
        tag_idx = info['idx']
        document_frequency = info['df']
        idf_vector[tag_idx] = log((N + smoothing) / (document_frequency + smoothing))

    // Apply IDF to each edge
    weighted_values = empty array on GPU
    FOR each position in col_indices_gpu:
        tag_idx = col_indices_gpu[position]
        weighted_values[position] = idf_vector[tag_idx]

    move weighted_values back to CPU
    RETURN weighted_values

FUNCTION compute_ppmi_weights_gpu(DT_binary):
    convert DT_binary to COO format
    extract row_indices, col_indices from COO

    // Move to GPU
    row_indices_gpu = move row_indices to GPU
    col_indices_gpu = move col_indices to GPU

    // Compute row degrees (number of tags per document)
    row_degrees = create zero vector of size N on GPU
    FOR each position in row_indices_gpu:
        row_id = row_indices_gpu[position]
        increment row_degrees[row_id] by 1

    // Compute column degrees (number of documents per tag)
    col_degrees = create zero vector of size V on GPU
    FOR each position in col_indices_gpu:
        col_id = col_indices_gpu[position]
        increment col_degrees[col_id] by 1

    // Compute total pairs
    total_pairs = 0
    FOR each row_id in range(N):
        total_pairs += row_degrees[row_id] * sum(row_degrees)

    // Compute PPMI for each edge
    ppmi_values = empty array on GPU
    FOR each position in row_indices_gpu:
        row_id = row_indices_gpu[position]
        col_id = col_indices_gpu[position]

        pmi = log(total_pairs / (row_degrees[row_id] * col_degrees[col_id]))
        ppmi_values[position] = maximum of (0, pmi)

    move ppmi_values back to CPU
    RETURN ppmi_values

FUNCTION row_normalize_gpu(rows, values, num_rows, norm_type):
    move rows and values to GPU

    IF norm_type == 'l2' THEN:
        // L2 normalization
        row_norms_squared = create zero vector of size num_rows on GPU

        FOR each position in values:
            row_id = rows[position]
            add values[position]^2 to row_norms_squared[row_id]

        row_norms = square_root of row_norms_squared

        FOR each position in values:
            row_id = rows[position]
            values[position] = values[position] / row_norms[row_id]

    ELSE IF norm_type == 'l1' THEN:
        // L1 normalization
        row_sums = create zero vector of size num_rows on GPU

        FOR each position in values:
            row_id = rows[position]
            add absolute_value(values[position]) to row_sums[row_id]

        FOR each position in values:
            row_id = rows[position]
            values[position] = values[position] / row_sums[row_id]

    move values back to CPU
    RETURN values

// Main execution
tag_vocab = CALL build_tag_vocabulary(doc_df, MIN_DF_TAG)
V = size of tag_vocab

DT_binary = CALL build_DT_binary(doc_df, tag_vocab)

// Binary weighting (no change)
DT_bin_values = extract values from DT_binary

// TF-IDF weighting
DT_tfidf_values = CALL compute_tfidf_weights_gpu(DT_binary, tag_vocab, TF_IDF_SMOOTH)

// PPMI weighting
DT_ppmi_values = CALL compute_ppmi_weights_gpu(DT_binary)

// Row normalization
IF NORM_TYPE_TAG == 'l2' THEN:
    rows, cols = extract row and col indices from DT_binary
    DT_bin_values = CALL row_normalize_gpu(rows, DT_bin_values, N, 'l2')
    DT_tfidf_values = CALL row_normalize_gpu(rows, DT_tfidf_values, N, 'l2')
    DT_ppmi_values = CALL row_normalize_gpu(rows, DT_ppmi_values, N, 'l2')

// Save matrices
CALL save_csr_as_triplets(DT_bin, 'DT_bin.parquet')
CALL save_csr_as_triplets(DT_tfidf, 'DT_tfidf.parquet')
CALL save_csr_as_triplets(DT_ppmi, 'DT_ppmi.parquet')
```

---

## STEP 4: Text View Construction (D-W BM25 Matrix)

```
// Load documents
doc_df = load parquet from 'doc_clean.parquet'

FUNCTION tokenize_text(text, stopwords):
    tokens = split text by whitespace
    filtered_tokens = empty list

    FOR each token in tokens:
        IF token NOT in stopwords AND length of token >= 2 THEN:
            append token to filtered_tokens

    RETURN filtered_tokens

FUNCTION build_word_vocabulary(documents, min_df, max_df):
    word_counter = empty dictionary
    N = number of documents

    // Count document frequency
    FOR each document in documents:
        tokens = CALL tokenize_text(document.text_all, STOPWORDS)
        unique_words = get unique words from tokens

        FOR each word in unique_words:
            increment word_counter[word] by 1

    // Filter by document frequency
    vocabulary = empty dictionary
    word_index = 0
    FOR each (word, count) in word_counter:
        IF min_df <= count <= max_df * N THEN:
            vocabulary[word] = {
                'idx': word_index,
                'df': count
            }
            increment word_index

    RETURN vocabulary

FUNCTION build_DW_count(documents, word_vocabulary):
    rows = empty list
    cols = empty list
    data = empty list

    FOR doc_idx FROM 0 TO number of documents - 1:
        tokens = CALL tokenize_text(documents[doc_idx].text_all, STOPWORDS)
        word_counts = empty dictionary

        // Count word frequencies in document
        FOR each token in tokens:
            IF token in word_vocabulary THEN:
                word_idx = word_vocabulary[token]['idx']
                increment word_counts[word_idx] by 1

        // Add entries to sparse matrix
        FOR each (word_idx, count) in word_counts:
            append doc_idx to rows
            append word_idx to cols
            append count to data

    DW_count = create CSR matrix with (data, (rows, cols))
    set shape to (N, W) where W = size of word_vocabulary
    RETURN DW_count

FUNCTION compute_bm25_weights_gpu(DW_count, word_vocabulary, k1, b):
    N = number of rows in DW_count

    // Compute document lengths
    document_lengths = empty array of size N
    FOR doc_id FROM 0 TO N-1:
        document_lengths[doc_id] = sum of DW_count[doc_id, :]

    average_doc_length = mean of document_lengths

    // Convert to COO and move to GPU
    convert DW_count to COO format
    extract row_indices, col_indices, frequencies from COO

    move row_indices, col_indices, frequencies to GPU

    // Build IDF vector
    W = size of word_vocabulary
    idf_vector = create zero vector of size W on GPU

    FOR each (word, info) in word_vocabulary:
        word_idx = info['idx']
        df = info['df']
        idf_vector[word_idx] = log((N - df + 0.5) / (df + 0.5) + 1.0)

    // Compute BM25 for each edge
    bm25_scores = empty array on GPU

    FOR each position in frequencies:
        row_id = row_indices[position]
        col_id = col_indices[position]

        term_freq = frequencies[position]
        doc_len = document_lengths[row_id]
        idf = idf_vector[col_id]

        numerator = term_freq * (k1 + 1)
        denominator = term_freq + k1 * (1 - b + b * doc_len / average_doc_length)

        bm25_scores[position] = idf * (numerator / denominator)

    move bm25_scores back to CPU
    RETURN bm25_scores

// Main execution
stopwords = set of common words: ['the', 'a', 'an', 'and', 'or', 'but', ...]

word_vocab = CALL build_word_vocabulary(doc_df, MIN_DF_WORD, MAX_DF_WORD)
W = size of word_vocab

DW_count = CALL build_DW_count(doc_df, word_vocab)

DW_bm25_values = CALL compute_bm25_weights_gpu(DW_count, word_vocab, BM25_K1, BM25_B)

// Create BM25 matrix
rows, cols = extract row and col indices from DW_count
DW_bm25 = create CSR matrix with (DW_bm25_values, (rows, cols))

// Row normalization
IF NORM_TYPE_TEXT == 'l2' THEN:
    DW_bm25_values = CALL row_normalize_gpu(rows, DW_bm25_values, N, 'l2')
    DW_bm25 = create CSR matrix with (DW_bm25_values, (rows, cols))

CALL save_csr_as_triplets(DW_bm25, 'DW_bm25.parquet')
```

---

## STEP 5: GPU Random Walk Generator (Type-Constrained D-X-D Walks)

```
FUNCTION sample_neighbor(node_idx, adjacency_matrix):
    // Get neighbors and edge weights for node_idx
    row = get row node_idx from adjacency_matrix
    neighbors = extract column indices from row
    weights = extract values from row

    IF neighbors is empty THEN:
        RETURN NULL

    // Normalize weights to probabilities
    total_weight = sum of weights
    probabilities = weights / total_weight

    // Sample one neighbor
    sampled_index = randomly select from neighbors with probabilities
    RETURN sampled_index

FUNCTION generate_walk(start_doc, DX_matrix, XD_matrix, walk_length):
    walk = [start_doc]
    current_type = 'D'  // Start at Document node
    current_node = start_doc

    FOR step FROM 1 TO walk_length - 1:
        IF current_type == 'D' THEN:
            // Sample a tag/word neighbor
            next_x = CALL sample_neighbor(current_node, DX_matrix)

            IF next_x is NULL THEN:
                BREAK  // No outgoing edges, terminate walk

            // Offset X node ID to avoid collision with document IDs
            walk_node_id = next_x + N
            append walk_node_id to walk

            current_node = next_x
            current_type = 'X'

        ELSE:  // current_type == 'X'
            // Sample a document neighbor
            next_d = CALL sample_neighbor(current_node, XD_matrix)

            IF next_d is NULL THEN:
                BREAK  // No outgoing edges, terminate walk

            append next_d to walk

            current_node = next_d
            current_type = 'D'

    RETURN walk

CLASS TorchWalkCorpus:
    FUNCTION __init__(DX_matrix, XD_matrix, walk_length, walks_per_node):
        this.DX = convert DX_matrix to CSR format
        this.XD = convert XD_matrix to CSR format
        this.walk_length = walk_length
        this.walks_per_node = walks_per_node
        this.N = number of documents

    FUNCTION __iter__():
        FOR doc_id FROM 0 TO this.N - 1:
            FOR walk_num FROM 1 TO this.walks_per_node:
                walk = CALL generate_walk(
                    doc_id,
                    this.DX,
                    this.XD,
                    this.walk_length
                )

                IF length of walk >= 2 THEN:
                    YIELD walk

// Main execution
// Load matrices
DT_matrix = CALL load_csr_from_triplets('DT_tfidf.parquet', (N, V_tag))
DW_matrix = CALL load_csr_from_triplets('DW_bm25.parquet', (N, W_word))

// Transpose for bidirectional access
TD_matrix = transpose of DT_matrix
WD_matrix = transpose of DW_matrix

// Create walk generators
tag_walk_corpus = new TorchWalkCorpus(
    DT_matrix,
    TD_matrix,
    WALK_LENGTH_TAG,
    WALKS_PER_NODE
)

text_walk_corpus = new TorchWalkCorpus(
    DW_matrix,
    WD_matrix,
    WALK_LENGTH_TEXT,
    WALKS_PER_NODE
)

// Save walk parameters
walk_params_tag = {
    'DX_path': 'DT_tfidf.parquet',
    'shape': (N, V_tag),
    'walk_length': WALK_LENGTH_TAG,
    'walks_per_node': WALKS_PER_NODE,
    'seed': SEED_FOR_WALKS
}

save walk_params_tag to 'walk_tag_DT.pickle'
save walk_params_text to 'walk_text_DW.pickle'
```

---

## STEP 6: WS-SGNS Training (Skip-Gram with Negative Sampling)

```
CLASS SGNS:
    FUNCTION __init__(vocab_size, embedding_dim):
        // Two embedding matrices
        this.input_emb = create embedding matrix of size (vocab_size, embedding_dim)
        this.output_emb = create embedding matrix of size (vocab_size, embedding_dim)

        // Initialize weights
        initialize this.input_emb.weight uniformly in range [-0.5/embedding_dim, 0.5/embedding_dim]
        initialize this.output_emb.weight to zeros

    FUNCTION forward(center_ids, positive_ids, negative_ids):
        // center_ids: [batch_size]
        // positive_ids: [batch_size]
        // negative_ids: [batch_size, num_negatives]

        // Get embeddings
        center_emb = this.input_emb[center_ids]  // [batch_size, dim]
        pos_emb = this.output_emb[positive_ids]  // [batch_size, dim]
        neg_emb = this.output_emb[negative_ids]  // [batch_size, num_negatives, dim]

        // Compute positive scores (dot products)
        pos_score = empty vector of size batch_size
        FOR i FROM 0 TO batch_size - 1:
            pos_score[i] = dot_product(center_emb[i], pos_emb[i])

        // Positive loss: -log(sigmoid(pos_score))
        pos_loss = mean of -log(sigmoid(pos_score) + epsilon)

        // Compute negative scores
        neg_score = empty matrix of size (batch_size, num_negatives)
        FOR i FROM 0 TO batch_size - 1:
            FOR k FROM 0 TO num_negatives - 1:
                neg_score[i, k] = dot_product(center_emb[i], neg_emb[i, k])

        // Negative loss: -sum(log(sigmoid(-neg_score)))
        neg_loss = mean of sum(-log(sigmoid(-neg_score) + epsilon), axis=1)

        total_loss = pos_loss + neg_loss
        RETURN total_loss

FUNCTION build_negative_sampling_distribution(DX_matrix, power):
    // Compute node degrees
    degrees = empty array of size vocab_size
    FOR node_id FROM 0 TO vocab_size - 1:
        degrees[node_id] = sum of DX_matrix[node_id, :]

    // Raise to power (typically 0.75)
    degrees_powered = degrees ** power

    // Normalize to probabilities
    probabilities = degrees_powered / sum(degrees_powered)
    RETURN probabilities

FUNCTION extract_pairs_from_walks(walk_corpus, window_size):
    pairs = empty list

    FOR each walk in walk_corpus:
        walk_length = length of walk

        FOR center_idx FROM 0 TO walk_length - 1:
            center_node = walk[center_idx]

            // Left context window
            FOR offset FROM 1 TO window_size:
                context_idx = center_idx - offset
                IF context_idx >= 0 THEN:
                    context_node = walk[context_idx]
                    append (center_node, context_node) to pairs

            // Right context window
            FOR offset FROM 1 TO window_size:
                context_idx = center_idx + offset
                IF context_idx < walk_length THEN:
                    context_node = walk[context_idx]
                    append (center_node, context_node) to pairs

    RETURN pairs

// Main training
// Load walk parameters
walk_params = load pickle from 'walk_tag_DT.pickle'

// Reconstruct walk corpus
DX_matrix = CALL load_csr_from_triplets(walk_params['DX_path'], walk_params['shape'])
XD_matrix = transpose of DX_matrix
walk_corpus = new TorchWalkCorpus(DX_matrix, XD_matrix, walk_params['walk_length'], walk_params['walks_per_node'])

// Build negative sampling distribution
vocab_size = N + maximum of (V_tag, W_word)
neg_dist = CALL build_negative_sampling_distribution(DX_matrix, power=0.75)

// Extract training pairs
pairs = CALL extract_pairs_from_walks(walk_corpus, WINDOW_SIZE)

// Initialize model
model = new SGNS(vocab_size, EMBEDDING_DIM)
move model to GPU device
optimizer = create Adam optimizer with learning_rate=LEARNING_RATE

// Training loop
FOR epoch FROM 1 TO EPOCHS:
    shuffle pairs randomly

    // Mini-batch training
    FOR batch_start FROM 0 TO length of pairs STEP BATCH_SIZE:
        batch_end = minimum of (batch_start + BATCH_SIZE, length of pairs)
        batch_pairs = pairs[batch_start : batch_end]

        // Separate centers and positives
        centers = empty list
        positives = empty list
        FOR each (center, positive) in batch_pairs:
            append center to centers
            append positive to positives

        // Sample negatives
        batch_size = length of centers
        negatives = empty matrix of size (batch_size, NEG_SAMPLES)
        FOR i FROM 0 TO batch_size - 1:
            FOR k FROM 0 TO NEG_SAMPLES - 1:
                negatives[i, k] = randomly sample from vocab_size using neg_dist

        // Convert to tensors and move to GPU
        center_tensor = create LongTensor from centers and move to GPU
        pos_tensor = create LongTensor from positives and move to GPU
        neg_tensor = create LongTensor from negatives and move to GPU

        // Forward pass
        loss = CALL model.forward(center_tensor, pos_tensor, neg_tensor)

        // Backward pass
        reset optimizer gradients to zero
        compute gradients via backpropagation on loss
        update model parameters using optimizer

    print "Epoch", epoch, "completed"

// Extract embeddings
WITH no gradient computation:
    all_embeddings = get weight matrix from model.input_emb
    doc_embeddings = all_embeddings[0 : N]  // First N rows are documents

    // L2 normalization
    FOR i FROM 0 TO N - 1:
        row_norm = square_root of sum(doc_embeddings[i, :]^2)
        doc_embeddings[i, :] = doc_embeddings[i, :] / (row_norm + epsilon)

save doc_embeddings to 'Z_tag.npy'
```

---

## STEP 7: ANN Graph Construction (FAISS K-NN)

```
FUNCTION build_knn_graph_faiss(embeddings, K, use_gpu):
    N = number of rows in embeddings
    D = number of columns in embeddings

    // Build FAISS index
    IF use_gpu THEN:
        create GPU resources
        index = create GPU inner product index with dimension D
    ELSE:
        index = create CPU inner product index with dimension D

    // Add embeddings to index
    CALL index.add(embeddings)

    // Search K nearest neighbors (request K+1 to include self)
    similarities = empty matrix of size (N, K+1)
    neighbor_ids = empty matrix of size (N, K+1)

    (similarities, neighbor_ids) = CALL index.search(embeddings, K + 1)

    // Build triplets (exclude self-loops)
    triplets = empty list

    FOR query_id FROM 0 TO N - 1:
        FOR k FROM 0 TO K:
            neighbor_id = neighbor_ids[query_id, k]
            similarity = similarities[query_id, k]

            // Skip self-loop
            IF neighbor_id == query_id THEN:
                CONTINUE

            append (query_id, neighbor_id, similarity) to triplets

    RETURN triplets as DataFrame

FUNCTION partition_triplets(triplets_df, num_parts):
    N = maximum row index in triplets_df + 1
    part_size = ceiling of (N / num_parts)

    manifests = empty list

    FOR part_id FROM 0 TO num_parts - 1:
        row_start = part_id * part_size
        row_end = minimum of ((part_id + 1) * part_size, N)

        // Filter triplets for this partition
        part_df = select rows from triplets_df WHERE row >= row_start AND row < row_end

        // Save partition
        filepath = "tag_knn_part" + part_id + ".parquet"
        save part_df to filepath

        // Record metadata
        manifest_entry = {
            'part_id': part_id,
            'filepath': filepath,
            'row_start': row_start,
            'row_end': row_end,
            'num_triplets': number of rows in part_df
        }
        append manifest_entry to manifests

    RETURN manifests as DataFrame

// Main execution
// Load embeddings
Z_tag = load numpy array from 'Z_tag.npy' as float32
Z_text = load numpy array from 'Z_text.npy' as float32

// Build K-NN graphs
tag_knn_triplets = CALL build_knn_graph_faiss(Z_tag, K_ANN, FAISS_GPU)
text_knn_triplets = CALL build_knn_graph_faiss(Z_text, K_ANN, FAISS_GPU)

// Partition and save
tag_manifest = CALL partition_triplets(tag_knn_triplets, K_PARTS)
save tag_manifest to 'tag_knn_manifest.parquet'

text_manifest = CALL partition_triplets(text_knn_triplets, K_PARTS)
save text_manifest to 'text_knn_manifest.parquet'
```

---

## STEP 8: Graph Symmetrization + Row Normalization

```
FUNCTION load_full_graph_from_partitions(manifest):
    all_triplets = empty list

    // Load all partitions
    FOR each row in manifest:
        filepath = row['filepath']
        part_df = load parquet from filepath
        append part_df to all_triplets

    // Combine all triplets
    combined = concatenate all_triplets

    // Build sparse matrix
    N = maximum row index in combined + 1
    rows = extract 'row' column from combined
    cols = extract 'col' column from combined
    data = extract 'data' column from combined

    csr = create CSR matrix with (data, (rows, cols)) and shape (N, N)
    RETURN csr

FUNCTION symmetrize_graph(sparse_matrix, method):
    IF method == 'max' THEN:
        // Element-wise maximum of A and A^T
        A_transpose = transpose of sparse_matrix
        A_sym = element_wise_maximum(sparse_matrix, A_transpose)

    ELSE IF method == 'avg' THEN:
        // Average of A and A^T
        A_transpose = transpose of sparse_matrix
        A_sum = sparse_matrix + A_transpose
        A_sym = A_sum / 2.0

    ELSE IF method == 'union' THEN:
        // Union of edges from A and A^T
        A_transpose = transpose of sparse_matrix

        convert sparse_matrix to COO format as coo1
        convert A_transpose to COO format as coo2

        combined_triplets = empty dictionary

        // Add edges from A
        FOR i FROM 0 TO number of entries in coo1 - 1:
            row = coo1.row[i]
            col = coo1.col[i]
            value = coo1.data[i]
            combined_triplets[(row, col)] = value

        // Add edges from A^T (if not already present)
        FOR i FROM 0 TO number of entries in coo2 - 1:
            row = coo2.row[i]
            col = coo2.col[i]
            value = coo2.data[i]
            IF (row, col) NOT in combined_triplets THEN:
                combined_triplets[(row, col)] = value

        // Build symmetric matrix
        rows = extract all keys[0] from combined_triplets
        cols = extract all keys[1] from combined_triplets
        data = extract all values from combined_triplets
        A_sym = create CSR matrix with (data, (rows, cols))

    RETURN A_sym

FUNCTION row_normalize_sparse(sparse_matrix):
    N = number of rows in sparse_matrix

    // Compute row sums
    row_sums = empty array of size N
    FOR i FROM 0 TO N - 1:
        row_sums[i] = sum of sparse_matrix[i, :]

    // Avoid division by zero
    FOR i FROM 0 TO N - 1:
        IF row_sums[i] < epsilon THEN:
            row_sums[i] = 1.0

    // Compute inverse of row sums
    inv_row_sums = 1.0 / row_sums

    // Create diagonal matrix D^{-1}
    D_inv = create diagonal matrix with inv_row_sums on diagonal

    // Row normalize: P = D^{-1} @ A
    P = D_inv * sparse_matrix

    RETURN P

// Main execution
// Load partitioned graphs
tag_manifest = load parquet from 'tag_knn_manifest.parquet'
text_manifest = load parquet from 'text_knn_manifest.parquet'

tag_graph = CALL load_full_graph_from_partitions(tag_manifest)
text_graph = CALL load_full_graph_from_partitions(text_manifest)

// Symmetrize
tag_graph_sym = CALL symmetrize_graph(tag_graph, SYMMETRY_METHOD)
text_graph_sym = CALL symmetrize_graph(text_graph, SYMMETRY_METHOD)

// Row normalize
tag_graph_normalized = CALL row_normalize_sparse(tag_graph_sym)
text_graph_normalized = CALL row_normalize_sparse(text_graph_sym)

// Convert to triplets
tag_triplets = convert tag_graph_normalized to DataFrame with [row, col, data]
text_triplets = convert text_graph_normalized to DataFrame with [row, col, data]

// Re-partition and save
tag_symrow_manifest = CALL partition_triplets(tag_triplets, K_PARTS)
save tag_symrow_manifest to 'tag_graph_symrow_manifest.parquet'

text_symrow_manifest = CALL partition_triplets(text_triplets, K_PARTS)
save text_symrow_manifest to 'text_graph_symrow_manifest.parquet'
```

---

## STEP 9: Multi-View Fusion (Tag + Text Adaptive Fusion)

```
FUNCTION compute_row_concentration_gpu(sparse_matrix):
    N = number of rows in sparse_matrix

    convert sparse_matrix to COO format
    extract row_indices, values from COO

    move row_indices and values to GPU

    // Compute squared values
    squared_values = values ** 2

    // Accumulate squared values per row
    concentrations = create zero vector of size N on GPU
    FOR each position in squared_values:
        row_id = row_indices[position]
        add squared_values[position] to concentrations[row_id]

    move concentrations back to CPU
    RETURN concentrations

FUNCTION scale_sparse_by_row_weights_gpu(sparse_matrix, row_weights):
    convert sparse_matrix to COO format
    extract row_indices, col_indices, values from COO

    move row_indices, col_indices, values to GPU
    move row_weights to GPU

    // Scale each value by its row's weight
    scaled_values = empty array of same size as values on GPU
    FOR each position in values:
        row_id = row_indices[position]
        scaled_values[position] = values[position] * row_weights[row_id]

    move row_indices, col_indices, scaled_values back to CPU
    create scaled sparse matrix with (scaled_values, (row_indices, col_indices))
    RETURN scaled sparse matrix

FUNCTION topk_per_row_sparse(sparse_matrix, k):
    N = number of rows in sparse_matrix

    triplets = empty list

    FOR row_id FROM 0 TO N - 1:
        // Get row entries
        row = get row row_id from sparse_matrix
        col_indices = extract column indices from row
        values = extract values from row

        // Sort by value (descending)
        sorted_positions = argsort values in descending order

        // Keep top-k
        FOR i FROM 0 TO minimum of (k - 1, length of values - 1):
            pos = sorted_positions[i]
            col_id = col_indices[pos]
            value = values[pos]

            IF value > 0 THEN:
                append (row_id, col_id, value) to triplets

    create sparse matrix from triplets
    RETURN sparse matrix

FUNCTION adaptive_fusion_gpu(graph_A, graph_B, top_k):
    N = number of rows in graph_A

    // Compute row concentrations
    concentration_A = CALL compute_row_concentration_gpu(graph_A)
    concentration_B = CALL compute_row_concentration_gpu(graph_B)

    // Compute adaptive weights (inverse concentration)
    alpha_A = empty array of size N
    alpha_B = empty array of size N

    FOR i FROM 0 TO N - 1:
        alpha_A[i] = 1.0 / (concentration_A[i] + epsilon)
        alpha_B[i] = 1.0 / (concentration_B[i] + epsilon)

    // Scale graphs by adaptive weights
    A_scaled = CALL scale_sparse_by_row_weights_gpu(graph_A, alpha_A)
    B_scaled = CALL scale_sparse_by_row_weights_gpu(graph_B, alpha_B)

    // Merge scaled graphs
    // Convert to dense for addition (or use sparse addition if available)
    A_dense = convert A_scaled to dense matrix on GPU
    B_dense = convert B_scaled to dense matrix on GPU
    fused_dense = A_dense + B_dense

    // Convert back to sparse
    fused_sparse = convert fused_dense to sparse matrix

    // Top-K sparsification per row
    fused_topk = CALL topk_per_row_sparse(fused_sparse, top_k)

    // L2 normalize each row
    fused_normalized = empty matrix of same shape
    FOR i FROM 0 TO N - 1:
        row = get row i from fused_topk
        values = extract values from row

        row_norm = square_root of sum(values ** 2)

        IF row_norm > epsilon THEN:
            normalized_values = values / row_norm
        ELSE:
            normalized_values = values

        set row i in fused_normalized to normalized_values

    RETURN fused_normalized

// Main execution
// Load both symmetrized graphs
tag_manifest = load parquet from 'tag_graph_symrow_manifest.parquet'
text_manifest = load parquet from 'text_graph_symrow_manifest.parquet'

tag_graph = CALL load_full_graph_from_partitions(tag_manifest)
text_graph = CALL load_full_graph_from_partitions(text_manifest)

// Apply adaptive fusion
fused_graph = CALL adaptive_fusion_gpu(tag_graph, text_graph, TOP_K_FUSION)

// Convert to triplets and partition
fused_triplets = convert fused_graph to DataFrame with [row, col, data]
fused_manifest = CALL partition_triplets(fused_triplets, K_PARTS)

save fused_manifest to 'S_tag_text_fused_manifest.parquet'
```

---

## STEP B1-B4: Behavior View Construction

```
// STEP B1: Align behavior data
behavior_df = read CSV from 'behavior_interactions.csv'
index_map = load parquet from 'index_map.parquet'

// Merge behavior data with document index
behavior_aligned = merge behavior_df with index_map ON item_id = id

// Aggregate interactions
interaction_counts = GROUP behavior_aligned BY (user_id, internal_idx) AND COUNT

// STEP B2: Build collaborative similarity (S_ids)
FUNCTION build_collaborative_similarity(interaction_counts, num_users, num_docs):
    // Build user-item matrix
    rows = extract user_id from interaction_counts
    cols = extract internal_idx from interaction_counts
    data = extract count from interaction_counts

    user_item_matrix = create CSR matrix with (data, (rows, cols))
    set shape to (num_users, num_docs)

    // Normalize by item norms for cosine similarity
    item_norms = empty array of size num_docs
    FOR item_id FROM 0 TO num_docs - 1:
        col = get column item_id from user_item_matrix
        item_norms[item_id] = square_root of sum(col ** 2)

    user_item_normalized = create empty matrix of same shape
    FOR item_id FROM 0 TO num_docs - 1:
        col = get column item_id from user_item_matrix
        IF item_norms[item_id] > epsilon THEN:
            normalized_col = col / item_norms[item_id]
        ELSE:
            normalized_col = col
        set column item_id in user_item_normalized to normalized_col

    // Compute item-item similarity: I×I = (U×I)^T @ (U×I)
    item_similarity = transpose(user_item_normalized) * user_item_normalized

    RETURN item_similarity

S_ids = CALL build_collaborative_similarity(interaction_counts, num_users, N)

// STEP B3: Build engagement similarity (S_eng)
FUNCTION compute_engagement_features(interaction_counts):
    engagement_vectors = empty matrix of size (N, num_features)

    FOR doc_id FROM 0 TO N - 1:
        // Extract engagement metrics for this document
        doc_interactions = filter interaction_counts WHERE internal_idx == doc_id

        total_interactions = sum of doc_interactions.count
        unique_users = count of unique user_id in doc_interactions

        // Create feature vector
        engagement_vectors[doc_id, 0] = total_interactions
        engagement_vectors[doc_id, 1] = unique_users
        // ... more features

    RETURN engagement_vectors

engagement_features = CALL compute_engagement_features(interaction_counts)

// Normalize features
FOR feature_idx FROM 0 TO num_features - 1:
    col = engagement_features[:, feature_idx]
    col_mean = mean of col
    col_std = standard_deviation of col
    engagement_features[:, feature_idx] = (col - col_mean) / (col_std + epsilon)

// Build K-NN graph on engagement features
S_eng_triplets = CALL build_knn_graph_faiss(engagement_features, K_ANN, FAISS_GPU)
S_eng = convert S_eng_triplets to sparse matrix

// Symmetrize and normalize
S_eng_sym = CALL symmetrize_graph(S_eng, 'max')
S_eng_normalized = CALL row_normalize_sparse(S_eng_sym)

// STEP B4: Fuse behavior views
S_beh = CALL adaptive_fusion_gpu(S_ids, S_eng_normalized, TOP_K_FUSION)

// Save
S_beh_triplets = convert S_beh to DataFrame
S_beh_manifest = CALL partition_triplets(S_beh_triplets, K_PARTS)
save S_beh_manifest to 'S_beh_manifest.parquet'
```

---

## STEP C: Three-View Total Fusion

```
// Load all three views
tag_text_manifest = load parquet from 'S_tag_text_fused_manifest.parquet'
beh_manifest = load parquet from 'S_beh_manifest.parquet'

S_tag_text = CALL load_full_graph_from_partitions(tag_text_manifest)
S_beh = CALL load_full_graph_from_partitions(beh_manifest)

// Apply adaptive fusion
S_final = CALL adaptive_fusion_gpu(S_tag_text, S_beh, TOP_K_FINAL)

// Final row normalization
S_final_normalized = CALL row_normalize_sparse(S_final)

// Convert to triplets and partition
final_triplets = convert S_final_normalized to DataFrame with [row, col, data]
final_manifest = CALL partition_triplets(final_triplets, K_PARTS)

save final_manifest to 'S_fused3_symrow_manifest.parquet'

// This is the FINAL RECOMMENDATION GRAPH
```

---

## VERIFY: Neighbor Quality Inspection

```
// Load final graph
final_manifest = load parquet from 'S_fused3_symrow_manifest.parquet'
S_final = CALL load_full_graph_from_partitions(final_manifest)

// Load document metadata
doc_df = load parquet from 'doc_clean.parquet'

// Sample documents
num_samples = 10
sample_ids = randomly select num_samples document IDs from range [0, N-1]

FOR each doc_id in sample_ids:
    // Get neighbors
    row = get row doc_id from S_final
    neighbor_ids = extract column indices from row
    similarities = extract values from row

    // Sort by similarity (descending)
    sorted_indices = argsort similarities in descending order

    top_k = 10
    top_neighbors = neighbor_ids[sorted_indices[0 : top_k]]
    top_scores = similarities[sorted_indices[0 : top_k]]

    // Display
    print "=== Document", doc_id, "==="
    print "Text:", doc_df[doc_id].text_all[0:100], "..."
    print "Tags:", doc_df[doc_id].tag_str
    print ""
    print "Top Neighbors:"

    FOR rank FROM 1 TO top_k:
        neighbor_id = top_neighbors[rank - 1]
        score = top_scores[rank - 1]
        neighbor_text = doc_df[neighbor_id].text_all[0:80]
        neighbor_tags = doc_df[neighbor_id].tag_str

        print rank, ". [Score:", score, "]", neighbor_text, "... | Tags:", neighbor_tags

    print ""
```

---

## Summary: Complete Pipeline Flow

```
RAW_CSV
    ↓
[STEP 2] Preprocessing
    ├→ Clean text (remove HTML, URLs, lowercase)
    ├→ Parse tags
    └→ Save doc_clean.parquet, index_map.parquet
    ↓
    ├─────────────────────────┬──────────────────────────┐
    ↓                         ↓                          ↓
[STEP 3] Tag View        [STEP 4] Text View       [STEP B1] Behavior Data
    ├→ Build D-T matrix      ├→ Tokenize text           ├→ Align with docs
    ├→ TF-IDF weighting      ├→ Build D-W matrix        └→ Create user-item matrix
    └→ PPMI weighting        └→ BM25 weighting               ↓
    ↓                         ↓                          [B2] Collaborative Similarity
[STEP 5] Random Walks    [STEP 5] Random Walks          ├→ Compute item-item from
    ├→ D→T→D walks           ├→ D→W→D walks                  user-item: S_ids
    └→ Save walk params      └→ Save walk params            ↓
    ↓                         ↓                          [B3] Engagement Similarity
[STEP 6] SGNS Training   [STEP 6] SGNS Training          ├→ Extract features
    ├→ Extract pairs         ├→ Extract pairs               ├→ K-NN on features: S_eng
    ├→ Train embeddings      ├→ Train embeddings            ↓
    └→ Save Z_tag.npy        └→ Save Z_text.npy         [B4] Behavior Fusion
    ↓                         ↓                              ├→ Fuse S_ids + S_eng
[STEP 7] FAISS K-NN      [STEP 7] FAISS K-NN               └→ Save S_beh
    ├→ Build index           ├→ Build index                  ↓
    ├→ Search neighbors      ├→ Search neighbors             |
    └→ Save tag_knn          └→ Save text_knn                |
    ↓                         ↓                               |
[STEP 8] Symmetrize      [STEP 8] Symmetrize                |
    ├→ Make undirected       ├→ Make undirected              |
    └→ Row normalize         └→ Row normalize                |
    ↓                         ↓                               |
    └─────────┬───────────────┘                               |
              ↓                                               |
    [STEP 9] Tag+Text Fusion                                 |
        ├→ Adaptive fusion                                   |
        └→ Save S_tag_text_fused ───────┬────────────────────┘
                                        ↓
                              [STEP C] Three-View Fusion
                                  ├→ Fuse S_tag_text + S_beh
                                  └→ Save S_fused3_symrow
                                        ↓
                                  [VERIFY] Quality Check
                                  └→ Sample and inspect neighbors
```
