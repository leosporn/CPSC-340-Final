Something needed to be in each folder!

Something needed to be in each folder!


saving and loading pytorch models: https://pytorch.org/tutorials/beginner/saving_loading_models.html

save:
torch.save(model, PATH)


load:
```
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```
