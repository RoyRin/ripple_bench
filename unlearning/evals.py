from unlearning import metrics, models
from unlearning import dual_use_facts_utils


def eval_on_wmdp(dataframe, model, tokenizer):
    batch_size = 5

    batches = metrics.prepare_data_wmdp(dataframe.iterrows(), batch_size)
    corrects = metrics.get_accuracy(model, tokenizer, batches, None,
                                    N=None)  # len(dataframe)

    print(f"Number of corrects: {corrects}")
    acc = sum(corrects) / len(corrects)
    print(f"WMDP Accuracy: {acc}")

    #bio question

    bio_question, bio_answer = dual_use_facts_utils.get_wmdp_question_answer(
        dataframe, 2)

    prompt_format = f"extract a stand-alone, specific fact using information from this question-answer pair: {bio_question} \nAnswer: {bio_answer} \nFact: "

    prompt_format = f"extract the subject of this question-answer pair: {bio_question} \nAnswer: {bio_answer} \nFact: "

    # ask the model
    out = models.generate_text(prompt_format, max_new_tokens=200)
    print(f"{out}")

    df_columns = dataframe.columns.tolist()

    dual_use_dataframe = pd.DataFrame(columns=df_columns)
    # add a row

    # add row
    row = dataframe.iloc[3]
    choices = row['choices']
    question = row['question']
    answer_ind = row['answer']
    choices

    dual_use_row = dual_use_facts_utils.construct_single_dual_use_df_row(
        question, choices)
    # pd.concat([dual_use_dataframe, row], ignore_index=True)
    return dual_use_row
