def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""  

    inputs = examples["mask_data"]
    en_labels = examples["en_labels"]
    targets = examples["label_data"]
    de_labels = examples["de_labels"]

    PAD_TOKEN_LABEL = -100
    PAD_LABEL_LABEL = 5 
    
    features = []
    for input,target,en_label,de_label in zip(inputs,targets,en_labels,de_labels):
        
        
        

        attention_mask=[1]*len(input)
        # print(len(input),len(en_label))
        # Zero-pad up to the sequence length.
        # print(len(input),input)
        while len(input) < max_seq_length:
            input.append(1)
            attention_mask.append(0)
            en_label.append(PAD_LABEL_LABEL)
            
        while len(target) < max_seq_length:
            target.append(PAD_TOKEN_LABEL)
            de_label.append(-100)

        assert len(input) == max_seq_length,print(input,tokenizer.decode(input))
        assert len(attention_mask) == max_seq_length
        assert len(target) == max_seq_length
        assert len(en_label) == max_seq_length


        # label_id = [PAD_TOKEN_LABEL] * len(input_ids)  # -1 is the index to ignore use 0
        # # truncate the label length if it exceeds the limit.
        # lb = [label_map[label] for label in labels_a]
        # if len(lb) > max_seq_length - 2:
        #     lb = lb[0:(max_seq_length - 2)]
        # label_id[1:len(lb) + 1] = lb  # 前后都是-1

        # label_id = [label_map[l] for l in labels_a]
        # label_padding = [-1] * (max_seq_length-len(label_id))
        # label_id += label_padding
        
        features.append(
            InputFeatures(
                input_ids=input,
                attention_mask=attention_mask,
                labels=target,
                en_labels=en_label,
                de_labels=de_label))

    return features
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, labels, en_labels, de_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.en_labels = en_labels
        self.de_labels = de_labels


def convert_examples_to_features_test(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""  

    inputs = examples["mask_data"]
    en_labels = examples["en_labels"]
    targets = examples["label_data"]
    de_labels = examples["de_labels"]

    PAD_TOKEN_LABEL = -100
    PAD_LABEL_LABEL = 5 
    
    features = []
    for input,target,en_label,de_label in zip(inputs,targets,en_labels,de_labels):
        
        
        

        attention_mask=[1]*len(input)
        # print(len(input),len(en_label))
        # Zero-pad up to the sequence length.
        while len(input) < max_seq_length:
            input.append(1)
            attention_mask.append(0)
            en_label.append(PAD_LABEL_LABEL)
            
        while len(target) < max_seq_length:
            target.append(PAD_TOKEN_LABEL)
            de_label.append(-100)

        assert len(input) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(target) == max_seq_length
        assert len(en_label) == max_seq_length


        # label_id = [PAD_TOKEN_LABEL] * len(input_ids)  # -1 is the index to ignore use 0
        # # truncate the label length if it exceeds the limit.
        # lb = [label_map[label] for label in labels_a]
        # if len(lb) > max_seq_length - 2:
        #     lb = lb[0:(max_seq_length - 2)]
        # label_id[1:len(lb) + 1] = lb  # 前后都是-1

        # label_id = [label_map[l] for l in labels_a]
        # label_padding = [-1] * (max_seq_length-len(label_id))
        # label_id += label_padding
        
        features.append(
            InputFeatures(
                input_ids=input,
                attention_mask=attention_mask,
                labels=target,
                en_labels=en_label,
                de_labels=de_label))

    return features