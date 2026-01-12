# example
import torch
import edge_detector as ce
from torchvision.io import read_image
from torchvision.transforms import v2
import matplotlib.pyplot as plt

tensor = read_image('./resources/example.jpeg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor= v2.Grayscale()(tensor)
tensor= tensor.unsqueeze(0)
tensor = tensor.to(torch.float16)
tensor = tensor.to(device)

# simulate batch size of 32

test= tensor.repeat(32,1,1,1)

getedge = ce.c_edge(upper_treshold = 40,lower_treshold = 20,max_iterations=15)

# regular
# getedge.to(device)
# jit
getedge = torch.jit.script(ce.c_edge(upper_treshold = 40,lower_treshold = 20))
getedge.to(device)

edges = getedge(tensor)


edges = edges[0].squeeze()
npedges = edges.cpu().numpy()
plt.imshow(npedges)

# onnx_program = torch.onnx.export(getedge, test, dynamo=True)
