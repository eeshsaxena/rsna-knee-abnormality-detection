import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ck["args"] = dict(ck["args"], train_labels="other_labels.csv")
torch.save(ck, sys.argv[2])
