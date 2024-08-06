def return_longest_spoiler_index(spoilers):
    max_len = 0
    for i,spoiler in enumerate(spoilers):
        if len(spoiler) > max_len:
            max_len = len(spoiler)
            longest_spoiler = i
    return longest_spoiler


def findPosTags(x):
    tokPos = []
    for pos in x['spoilerPositions']:
        st,en = pos
        idx = 0
        for i,p in enumerate([x['targetTitle']] + x['targetParagraphs']):
            if i==st[0]+1:
                start_ind = idx + st[1]
                end_ind = idx + en[1]

                tokPos.append([start_ind,end_ind])
                break
            if i==0:
                idx+=len(p)+3
            else:
                idx+=len(p)+1

    return tokPos

def convert2squadFormat(df, dataset_name):
    if dataset_name == 'test':
        df_fin = df[['id','targetTitle','postText',"title_para"]]
        df_fin['id'] = df_fin['id'].astype(str)
        df_fin.columns = ['id',"title","question","context"]
    else:
        df_fin = df[['id','label','targetTitle','postText',"title_para","longest_tokPos","longest_spoiler"]]
        df_fin["asnwers"] = df_fin.apply(lambda x: {'text':[x['longest_spoiler']], "answer_start":[x['longest_tokPos'][0]]},1)
        df_fin = df_fin.drop(columns=["longest_tokPos","longest_spoiler"])
        df_fin['id'] = df_fin['id'].astype(str)
        df_fin.columns = ['id','labels',"title","question","context","answers"]

    return df_fin

def preprocess_training_examples(examples):
    questions = examples['question']
    tags = examples['labels']
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    final_tags = []

    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        tag = tags[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        final_tags.append(tag)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs['labels'] = final_tags
    return inputs


def compute_metrics(start_logits, end_logits,class_logits, features, examples, predictOnly=False):
    example_to_features = defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_spoilers = []
    predicted_classes = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        spoilers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        'class': class_logits[feature_index].argmax()
                    }
                    spoilers.append(answer)

        # Select the answer with the best score
        if len(spoilers) > 0:
            best_answer = max(spoilers, key=lambda x: x["logit_score"])
            predicted_spoilers.append(
                {"id": example_id, "prediction_text": best_answer["text"]})
            predicted_classes.append(best_answer['class'])
        else:
            predicted_spoilers.append({"id": example_id, "prediction_text": ""})
            predicted_classes.append(0)

    predicted_texts = [i['prediction_text'] for i in predicted_spoilers]

    if predictOnly:
        return predicted_texts, predicted_classes

    actual_spoilers_squad = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    actual_spoilers = [i['answers']['text'][0] for i in actual_spoilers_squad]
    actual_class = [ex['labels'] for ex in examples]

    squad_metrics_eval = squad_metric.compute(predictions=predicted_spoilers, references=actual_spoilers_squad)
    bleu_eval = bleu.compute(predictions=predicted_texts, references=actual_spoilers)
    meteor_eval = meteor.compute(predictions=predicted_texts, references=actual_spoilers)
    rouge_eval = rouge.compute(predictions=predicted_texts, references=actual_spoilers)
    class_f1 = f1_score(actual_class, predicted_classes, average='weighted')
    class_acc = accuracy_score(actual_class, predicted_classes)
    other_metrics = {'f1':class_f1, 'accuracy':class_acc}

    metrics = {
        'squad_exact_match':squad_metrics_eval['exact_match'],
        'squad_f1':squad_metrics_eval['f1'],
        'bleu':bleu_eval['bleu'],
        'meteor':meteor_eval['meteor'],
        'rouge1':rouge_eval['rouge1'],
        'rouge2':rouge_eval['rouge2'],
        'rougeL':rouge_eval['rougeL'],
        'rougeLsum':rouge_eval['rougeLsum'],
        'class_accuracy':class_acc,
        'class_f1':class_f1
        }

    for k,v in metrics.items():
        metrics[k] = np.round(v, 4)

    return metrics, actual_spoilers,predicted_texts, predicted_classes

#    return [squad_metrics_eval,bleu_eval,meteor_eval,rouge_eval, other_metrics],actual_spoilers,predicted_texts, predicted_classes

def preprocess_test_examples(examples):
    questions = examples["question"]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    print('len(inputs["input_ids"])',len(inputs["input_ids"]))

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids

    return inputs

def fixFormat(df, dataset_name):
    if dataset_name == 'test':
        df_fin = df[['id',"title_para"]]
        df_fin['id'] = df_fin['id'].astype(str)
    else:
        df_fin = df[['id','label',"title_para"]]
        df_fin['id'] = df_fin['id'].astype(str)

    return df_fin

def preprocess_training_examples_cls(examples):
    text = examples['title_para']
    tags = examples['label']
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")

    final_tags = []
    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = sample_map[i]
        tag = tags[sample_idx]
        final_tags.append(tag)

    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    inputs['labels'] = final_tags
    return inputs

def preprocess_test_examples_cls(examples):
    inputs = tokenizer(
        examples["title_para"],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    print('len(inputs["input_ids"])',len(inputs["input_ids"]))

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids

    return inputs


def compute_metrics_cls(start_logits, end_logits,class_logits, features, examples, predictOnly=False):
    example_to_features = defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_spoilers = []
    predicted_classes = []
    for example in tqdm(examples):
        example_id = example["id"]
        spoilers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            answer = {
                "logit_score": class_logits[feature_index].max(),
                'class': class_logits[feature_index].argmax()
            }
            spoilers.append(answer)

        # Select the answer with the best score
        if len(spoilers) > 0:
            best_answer = max(spoilers, key=lambda x: x["logit_score"])
            predicted_classes.append(best_answer['class'])
        else:
            predicted_classes.append(0)

    if predictOnly:
        return predicted_classes

    actual_class = [ex['label'] for ex in examples]

    class_f1 = f1_score(actual_class, predicted_classes, average='weighted')
    class_acc = accuracy_score(actual_class, predicted_classes)
    other_metrics = {'f1':class_f1, 'accuracy':class_acc}

    metrics = {
        'class_accuracy':class_acc,
        'class_f1':class_f1
        }

    for k,v in metrics.items():
        metrics[k] = np.round(v, 4)

    return metrics, predicted_classes

#    return [squad_metrics_eval,bleu_eval,meteor_eval,rouge_eval, other_metrics],actual_spoilers,predicted_texts, predicted_classes


def convert2squadFormat_qa(df, dataset_name):
    if dataset_name == 'test':
        df_fin = df[['id','targetTitle','postText',"title_para"]]
        df_fin['id'] = df_fin['id'].astype(str)
        df_fin.columns = ['id',"title","question","context"]
    else:
        df_fin = df[['id','label','targetTitle','postText',"title_para","longest_tokPos","longest_spoiler"]]
        df_fin["asnwers"] = df_fin.apply(lambda x: {'text':[x['longest_spoiler']], "answer_start":[x['longest_tokPos'][0]]},1)
        df_fin = df_fin.drop(columns=["longest_tokPos","longest_spoiler"])
        df_fin['id'] = df_fin['id'].astype(str)
        df_fin.columns = ['id','labels',"title","question","context","answers"]

    return df_fin

def preprocess_training_examples_qa(examples):
    questions = examples['question']
    tags = examples['labels']
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    final_tags = []

    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        tag = tags[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        final_tags.append(tag)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def compute_metrics_qa(start_logits, end_logits, features, examples, predictOnly=False):
    example_to_features = defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_spoilers = []
    predicted_classes = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        spoilers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index]
                    }
                    spoilers.append(answer)

        # Select the answer with the best score
        if len(spoilers) > 0:
            best_answer = max(spoilers, key=lambda x: x["logit_score"])
            predicted_spoilers.append(
                {"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_spoilers.append({"id": example_id, "prediction_text": ""})


    predicted_texts = [i['prediction_text'] for i in predicted_spoilers]

    if predictOnly:
        return predicted_texts

    actual_spoilers_squad = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    actual_spoilers = [i['answers']['text'][0] for i in actual_spoilers_squad]
    actual_class = [ex['labels'] for ex in examples]

    squad_metrics_eval = squad_metric.compute(predictions=predicted_spoilers, references=actual_spoilers_squad)
    bleu_eval = bleu.compute(predictions=predicted_texts, references=actual_spoilers)
    meteor_eval = meteor.compute(predictions=predicted_texts, references=actual_spoilers)
    rouge_eval = rouge.compute(predictions=predicted_texts, references=actual_spoilers)

    metrics = {
        'squad_exact_match':squad_metrics_eval['exact_match'],
        'squad_f1':squad_metrics_eval['f1'],
        'bleu':bleu_eval['bleu'],
        'meteor':meteor_eval['meteor'],
        'rouge1':rouge_eval['rouge1'],
        'rouge2':rouge_eval['rouge2'],
        'rougeL':rouge_eval['rougeL'],
        'rougeLsum':rouge_eval['rougeLsum'],
        }

    for k,v in metrics.items():
        metrics[k] = np.round(v, 4)

    return metrics, actual_spoilers,predicted_texts

#    return [squad_metrics_eval,bleu_eval,meteor_eval,rouge_eval, other_metrics],actual_spoilers,predicted_texts, predicted_classes

def preprocess_test_examples_qa(examples):
    questions = examples["question"]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    print('len(inputs["input_ids"])',len(inputs["input_ids"]))

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids

    return inputs