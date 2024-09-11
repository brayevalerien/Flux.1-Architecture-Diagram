# Flux.1-Architecture-Diagram
A detailed diagram laying out the full Flux.1 architecture as shared by Black Forest Labs.
![diagram](./flux_architecture_diagram.png)

(open the file to view it zoomed in)

# Road map
- [x] first draft using draw.io
- [ ] convert to TikZ, either on a single page or split into sections
- [ ] add tensor shapes annotations to all connections
- [ ] write small document to explain the key points of the architecture

# References
- [Flux.1 source code](https://github.com/black-forest-labs/flux)
- [@nrehiew original diagram](https://x.com/nrehiew_/status/1820404554795802684) (latter [reposted on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1ekr7ji/fluxs_architecture_diagram_dont_think_theres_a/) by u/pppodong)
- Tensor shapes calculations: [Pytorch documentation](https://pytorch.org/docs/stable/index.html)
- Original paper that inspired the SingleStream blocks: [Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*, 2023](https://arxiv.org/abs/2302.05442)