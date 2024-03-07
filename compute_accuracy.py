import matplotlib.pyplot as plt

def top_1_accuracy():
    total_lines = 9898
    correct_predictions = 0
    with open('formatted_top1_predictions.txt', 'r') as f:
        for line in f:
            first_list = [item.strip() for item in line.split(',')]
            img_path = first_list[0]
            prediction = first_list[1]
            solution = img_path.split('/')[2]
            if solution == prediction:
                correct_predictions += 1
    accuracy = correct_predictions / total_lines * 100
    message = f'Out of {total_lines} images, the correct class was predicted with {accuracy:.4f}% accuracy'
    print(message)
    with open('Top1_Result.txt', 'w') as f:
        f.write(message)

def top_5_accuracy():
    total_lines = 9898
    correct_predictions = 0
    first_guess = 0
    second_guess = 0
    third_guess = 0
    fourth_guess = 0
    fifth_guess = 0
    with open('formatted_top5_predictions.txt', 'r') as f:
        for line in f:
            first_list = [item.strip() for item in line.split(',')]
            predictions = first_list[1:]
            img_path = first_list[0]
            solution = img_path.split('/')[2]
            if solution in predictions:
                correct_predictions += 1
            if solution == predictions[0]:
                first_guess += 1
            elif solution == predictions[1]:
                second_guess += 1
            elif solution == predictions[2]:
                third_guess += 1
            elif solution == predictions[3]:
                fourth_guess += 1
            elif solution == predictions[4]:
                fifth_guess += 1
    percent_correct = correct_predictions / total_lines * 100
    missed = (total_lines - correct_predictions) / total_lines * 100
    percent_first = first_guess / total_lines * 100
    percent_second = second_guess / total_lines * 100
    percent_third = third_guess / total_lines * 100
    percent_fourth = fourth_guess / total_lines * 100
    percent_fifth = fifth_guess / total_lines * 100
    message = f'Out of {total_lines} images, {correct_predictions} ({percent_correct:.2f}%) images\' classes were predicted accurately within 5 guesses. Out of all {total_lines}:\n{percent_first:.2f}% were predicted on the first try.\n{percent_second:.2f}% were predicted on the second try.\n{percent_third:.2f}% were predicted on the third try.\n{percent_fourth:.2f}% were predicted on the fourth try.\n{percent_fifth:.2f}% were predicted on the fifth try.\n{100 - (percent_first + percent_second + percent_third + percent_fourth + percent_fifth):.2f}% were missed.'
    print(message)
    with open('Top5_Result.txt', 'w') as f:
        f.write(message)
    categories = ['Total Correct','First Guess', 'Second Guess', 'Third Guess', 'Fourth Guess', 'Fifth Guess', 'Missed']
    percentages = [percent_correct, percent_first, percent_second, percent_third, percent_fourth, percent_fifth, missed]
    colors = ['green' if category != 'Missed' else 'red' for category in categories]
    bars = plt.bar(categories, percentages, color=colors)
    plt.title('Model Top-5 Accuracy Distribution')
    plt.xlabel('Guess')
    plt.ylabel('Percentage of Correct Answers')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
            

if __name__ == "__main__":
    top_1_accuracy()
    top_5_accuracy()

