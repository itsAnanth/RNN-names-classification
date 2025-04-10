import torch
import torch.utils
import torch.utils.data
from model import RNN

def train(model: RNN, criterion, optimizer, epochs, data, losses = []):
    
    running_loss = 0
    i = 0
    while True:
        i += 1
        optimizer.zero_grad()
        word, category = data.random_sample()
        hidden = model.init_hidden()

        for ch in range(word.shape[0]):
            output, hidden = model(word[ch], hidden)

    
        loss = criterion(output, category)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i % 1000 == 0):
            guess = data.tensor2category(category.item())
            correct = data.tensor2category(torch.argmax(output))
            # check = '✓' if guess == category else f'✗ ({category})'
            # print(f"Actual: {correct} | Predicted: {guess} {check}")
        
        if (i % 1000 == 0):
            print(f"Iterations: {i} | loss: {running_loss / 1000}")
            losses.append(running_loss / 1000)
            running_loss = 0
            
            
        running_loss += loss.item()
        