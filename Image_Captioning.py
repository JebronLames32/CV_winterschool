import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.datasets import CocoCaptions
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider

image_path = './image.jpg'

# Step 1: Dataset Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CocoCaptions(root='/path/to/train/dataset',
                             annFile='/path/to/annotations/train/annotations.json',
                             transform=transform)

val_dataset = CocoCaptions(root='/path/to/val/dataset',
                           annFile='/path/to/annotations/val/annotations.json',
                           transform=transform)

# Step 2: Data Preprocessing
def build_vocab(dataset):
    captions = [caption for _, caption in dataset.coco.anns.items()]
    word_freq = {}
    for caption in captions:
        for word in caption.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = [word for word, freq in word_freq.items() if freq >= min_word_freq]
    vocab.insert(0, '<start>')
    vocab.append('<end>')
    vocab.append('<unk>')
    return vocab

def tokenize_captions(dataset, vocab):
    dataset.vocab = vocab
    dataset.word2idx = {word: idx for idx, word in enumerate(vocab)}
    dataset.idx2word = {idx: word for idx, word in enumerate(vocab)}

    for ann_id, ann in dataset.coco.anns.items():
        caption = ann['caption']
        tokens = []
        tokens.append(dataset.word2idx['<start>'])
        tokens.extend([dataset.word2idx.get(word, dataset.word2idx['<unk>']) for word in caption.lower().split()])
        tokens.append(dataset.word2idx['<end>'])
        dataset.coco.anns[ann_id]['tokens'] = tokens

# Step 3: Model Architecture
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.rnn(packed)
        outputs = self.fc(hiddens[0])
        return outputs

# Step 4: Model Training
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        optimizer.zero_grad()
        outputs = model(images, captions, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Step 5: Inference
def generate_caption(model, image, max_length, dataset):
    model.eval()
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        features = model.cnn(image)
        inputs = torch.tensor([dataset.word2idx['<start>']], device=device).unsqueeze(0)

        caption = []

        for _ in range(max_length):
            embeddings = model.embedding(inputs)
            embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
            hiddens, _ = model.rnn(embeddings)
            outputs = model.fc(hiddens.squeeze(0))
            _, predicted = outputs.max(1)
            predicted_word = dataset.idx2word[predicted.item()]
            caption.append(predicted_word)

            if predicted_word == '<end>':
                break

            inputs = predicted.unsqueeze(0)

    return ' '.join(caption)

# Step 6: Evaluation
def evaluate(model, val_loader, dataset):
    model.eval()
    references = []
    hypotheses = []

    for i, (images, captions, _) in enumerate(val_loader):
        images = images.to(device)

        with torch.no_grad():
            captions = captions[:, 1:]
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            captions = captions.to(device)
            targets = targets.to(device)

            outputs = model(images, captions, lengths)
            _, predicted = outputs.max(1)
            predicted = predicted.tolist()

        references.extend([[caption[1:-1].tolist()] for caption in captions])
        hypotheses.extend(predicted)

    bleu_score = corpus_bleu(references, hypotheses)
    cider_score = compute_cider_score(references, hypotheses, dataset)

    return bleu_score, cider_score

def compute_cider_score(references, hypotheses, dataset):
    cider_scorer = Cider()
    gts = {}
    res = {}

    for i, ref in enumerate(references):
        img_id = dataset.ids[i]
        gts[img_id] = ref

    for i, hyp in enumerate(hypotheses):
        img_id = dataset.ids[i]
        res[img_id] = hyp

    _, cider_scores = cider_scorer.compute_score(gts, res)
    return cider_scores

# Hyperparameters
embed_size = 256
hidden_size = 512
num_epochs = 10
batch_size = 64
learning_rate = 0.001
min_word_freq = 5
max_caption_length = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 2: Data Preprocessing
vocab = build_vocab(train_dataset)
tokenize_captions(train_dataset, vocab)
tokenize_captions(val_dataset, vocab)

# Step 3: Model Architecture
model = ImageCaptioningModel(embed_size, hidden_size, len(vocab)).to(device)

# Step 4: Model Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

# Step 5: Inference
test_image = Image.open(image_path)
test_image = transform(test_image).unsqueeze(0)

caption = generate_caption(model, test_image, max_caption_length, train_dataset)



print('Generated Caption:', caption)

# Step 6: Evaluation
val_loader = DataLoader(val_dataset, batch_size=batch_size)
bleu_score, cider_score = evaluate(model, val_loader, val_dataset)
print(f'BLEU-4 Score: {bleu_score:.4f}')
print(f'CIDEr Score: {cider_score:.4f}')
