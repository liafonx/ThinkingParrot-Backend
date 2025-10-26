import re
import sys

from django.http import JsonResponse

# sys.path.append('/home/Ubuntu-UIC/ThinkingParrot-Backend/Chatbot')
sys.path.append('/Users/liafo/Documents/GitWorkspace/ThinkingParrot-Backend/Chatbot')
# from miniproject import chatbot
from Chatbot import TextPreprocessing, InputProcessing


def ChatbotGetMessage(request):
    message = request.POST.get("message")
    print(request)
    input_sentence = TextPreprocessing.normalizeString(message)
    output_words = InputProcessing.evaluate(input_sentence)

    outword = []
    for j in output_words:
        if j == 'EOS':
            break
        elif j != 'PAD':
            outword.append(j)
    string = ' '.join(outword)
    string = re.sub(' ll ', "'ll ", string)
    string = re.sub(' t ', "'t ", string)
    string = re.sub(' d ', "'d ", string)
    string = re.sub(' re ', "'re ", string)
    string = re.sub(' s ', "'s ", string)
    string = re.sub(' m ', " am ", string)
    string = re.sub(' ve ', "'ve ", string)
    # out_dict[i] = string

    # for j in input_list:
    #     print("Human :", j)
    #     print("Bot   :", out_dict[j])

    # reply = ' '.join(string).strip()
    return JsonResponse({'state': 'success', "response": string})
