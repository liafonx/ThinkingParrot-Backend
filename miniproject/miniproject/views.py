# Lib for Preprocessing and load data

from io import open
import re
import unicodedata
from colorama import Style, Fore
from django.http import JsonResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

# USE_CUDA = torch.cuda.is_available()
# from miniproject.miniproject.Chatbot.LeeOscillator import LeeOscillator

USE_CUDA = False
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.manual_seed(1)

MAX_LENGTH = 10  # Maximum sentence length to consider

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

hidden_size = 1536


class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS",
                           EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1


class TextPreprocessing:
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def normalizeString(s):
        s = TextPreprocessing.unicodeToAscii(s.lower().strip())
        s = re.sub(r"[^a-z]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def readVocs(datafile, corpus_name):
        lines = open(datafile, encoding='utf-8').read().strip().split('\n')
        pairs = [[TextPreprocessing.normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = Voc(corpus_name)
        return voc, pairs

    def filterPair(p):
        return (len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH)

    def filterPairs(pairs):
        return [pair for pair in pairs if TextPreprocessing.filterPair(pair)]

    def dropNull(pairs):
        return [pair for pair in pairs if pair[0] != '' and pair[1] != '']

    def loadPrepareData(corpus_name, datafile):
        voc, pairs = TextPreprocessing.readVocs(datafile, corpus_name)
        pairs = TextPreprocessing.filterPairs(pairs)
        pairs = TextPreprocessing.dropNull(pairs)
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        return voc, pairs


class DecoderPredict(nn.Module):
    def __init__(self, encoder, decoder):
        super(DecoderPredict, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # print("In: DecoderPredict, encoder", encoder_outputs.size())
        decoder_hidden = encoder_hidden[:1]
        decoder_input = torch.ones(1, 1, device=device,
                                   dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


class DataAdjustment:
    def tokenization(voc, sentence):
        return [voc['word2index'][word] for word in sentence.split(' ')] + [EOS_token]

    def zeroPadding(l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def dataVar(l, voc, con=True):
        indexes_batch = [DataAdjustment.tokenization(voc, sentence) for sentence in l]
        padList = DataAdjustment.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        if con:
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            return padVar, lengths
        else:
            max_target_len = max([len(indexes) for indexes in indexes_batch])
            return padVar, max_target_len

    def adjustBatchData(voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = DataAdjustment.dataVar(input_batch, voc)
        output, max_target_len = DataAdjustment.dataVar(output_batch, voc, False)
        return inp, lengths, output, max_target_len

    def batching(batch_size, iterable):
        args = [iter(iterable)] * batch_size
        return ([e for e in t if e != None] for t in itertools.zip_longest(*args))


# Lee = LeeOscillator()


class EncoderGRU(nn.Module):
    def __init__(self, hidden_size, embedding):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          bidirectional=True)
        # self.gru = ChaoticGRU(hidden_size, hidden_size, Lee, True, True)

    #         # self.gru2 = ChaoticGRU(hidden_size, hidden_size, Lee, True, True)
    #         self.rnn1 = ChaoticLSTM(hidden_size, hidden_size, Lee, True, True)
    #         self.rnn2 = ChaoticLSTM(hidden_size, hidden_size, Lee, True, True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        # print("In: EncoderGRU, embedded", embedded.size())
        # packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(embedded, hidden)
        #         outputs1, hidden1 = self.rnn1(embedded, initStates=hidden)
        #         # print("In: EncoderGRU, outputs_1*", outputs1.size(), " hidden: ", hidden1.size())
        #         outputs, hidden2 = self.rnn2(embedded, initStates=hidden1)
        #         (hidden1, _) = hidden1
        #         (hidden2, _) = hidden2
        #         hidden = torch.cat([hidden1, hidden2], dim=0)
        # hidden: [batch_size, 2, hidden_size]
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # print("In: EncoderGRU, outputs_1", outputs2.size())
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # print("In: EncoderGRU, outputs_2", outputs2.size(), " hidden: ", hidden.size())
        return outputs, hidden


class AttnDecoderGRU(nn.Module):
    def __init__(self, embedding, hidden_size,
                 output_size):
        super(AttnDecoderGRU, self).__init__()
        # decoder_input, decoder_hidden, encoder_outputs
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Define layers
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          bidirectional=False)
        # self.gru = ChaoticGRU(hidden_size, hidden_size, Lee, True, False)
        # self.gru2 = ChaoticGRU(hidden_size, hidden_size, Lee, True, False)
        #         self.rnn1 = ChaoticLSTM(hidden_size, hidden_size, Lee, True, False)
        #         self.rnn2 = ChaoticLSTM(hidden_size, hidden_size, Lee, True, False)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # print("In: AttnDecoderGRU, input_step", input_step.size())
        embedded = self.embedding(input_step)
        # print("In: AttnDecoderGRU, embedded", embedded.size(), "hidden: ", last_hidden.size())
        # Forward through unidirectional GRU
        #         h = last_hidden[-1:]
        #         h = torch.cat(h.split(1), dim=-1).squeeze(0)
        #         c = torch.zeros(h.size()).to(h.device)
        #         current_output1, current_hidden1 = self.rnn1(embedded, initStates=(h, c))
        #         gru_output, current_hidden2 = self.rnn2(embedded, initStates=current_hidden1)
        #         (current_hidden1, _) = current_hidden1
        #         (current_hidden2, _) = current_hidden2
        #         hidden = torch.cat([current_hidden1, current_hidden2], dim=0)
        gru_output, hidden = self.gru(embedded, last_hidden)
        # hidden = hidden.unsqueeze(0)
        # print("In: AttnDecoderGRU, gru_output", gru_output.size(), "hidden: ", hidden.size())
        # Calculate attention weights from the current GRU output
        luong_dot_score = torch.sum(gru_output * encoder_outputs, dim=2)
        attn_energies = luong_dot_score.t()
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        gru_output = gru_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((gru_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # print("In: AttnDecoderGRU, output", output.size(), " hidden: ", hidden.size())
        return output, hidden


# if __name__ == "__main__":
load_path = '/home/Ubuntu-UIC/ThinkingParrot-Backend/miniproject/miniproject/Chatbot/31536-512-50-0.0001_3.tar'
checkpoint = torch.load(load_path, map_location=torch.device("cpu"))

encoder_state_dict = checkpoint["en"]
decoder_state_dict = checkpoint["de"]
# print(encoder_state_dict)
embedding_state_dict = checkpoint["embedding"]
voc = checkpoint["voc_dict"]
# print(voc)
embedding = torch.nn.Embedding(num_embeddings=voc["num_words"],
                               embedding_dim=hidden_size)
embedding.load_state_dict(embedding_state_dict)

encoder = EncoderGRU(hidden_size=hidden_size, embedding=embedding)

decoder = AttnDecoderGRU(embedding=embedding, hidden_size=hidden_size, output_size=voc["num_words"])

decoder.load_state_dict(decoder_state_dict)
encoder.load_state_dict(encoder_state_dict)

encoder.eval()
decoder.eval()

# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

searcher = DecoderPredict(encoder, decoder)


class InputProcessing:
    def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
        indexes_batch = [DataAdjustment.tokenization(voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(device)
        lengths = lengths.to("cpu")
        tokens, scores = searcher(input_batch, lengths, max_length)
        decoded_words = [voc['index2word'][token.item()] for token in tokens]
        return decoded_words


def response_only(input_sentence):
    input_sentence = TextPreprocessing.normalizeString(input_sentence)
    output_words = InputProcessing.evaluate(encoder, decoder, searcher, voc, input_sentence)
    outword = []
    for i in output_words:
        if i == 'EOS' or i == 'PAD':
            break
        else:
            outword.append(i)
    return ' '.join(outword)

    # searcher = DecoderPredict(encoder, decoder)

    while True:
        user_dialog = input(f" {Fore.YELLOW}User:{Style.RESET_ALL} ").lower()
        if user_dialog == "q" or user_dialog == "quit":
            break
        # word转index， 并将不在词表中的单词替换成unk_token
        input_sentence = TextPreprocessing.normalizeString(user_dialog)
        output_words = InputProcessing.evaluate(encoder, decoder, searcher, voc, input_sentence)

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
        print(Fore.BLUE, "chatbot:", Style.RESET_ALL, string)


def ChatbotGetMessage(request):
    message = request.POST.get("message")
    print(request)
    input_sentence = TextPreprocessing.normalizeString(message)
    output_words = InputProcessing.evaluate(encoder, decoder, searcher, voc, input_sentence)

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
