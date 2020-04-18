# VAE Template
Use this as a



## How to use

To use in project or as a feature extractor
```python
from pytorch_lightning_bolts.models.vaes import VAE
import pytorch_lightning as pl

class YourResearchModel(pl.LightningModule):
    def __init__(self):
        self.vae = VAE.load_from_checkpoint(PATH)
        self.vae.freeze()

        self.some_other_model = MyModel()

    def forward(self, z):
        # generate a sample from z ~ N(0,1)
        x = self.vae(z)
        
        # do stuff with sample
        x = self.some_other_model(x)
        return x
```

To use in production or for predictions 
```python
from pytorch_lightning_bolts.models.vaes import VAE

vae = VAE.load_from_checkpoint(PATH)
vae.freeze()

z = ... # z ~ N(0, 1)
predictions = vae(z)
```

To train the VAE on its own
```python
from pytorch_lightning_bolts.models.vaes import VAE
import pytorch_lightning as pl

vae = VAE()
trainer = pl.Trainer(gpus=1)
trainer.fit(vae)
```

Train the VAE from the command line

```bash
cd pytorch_lightning_bolts/models/vaes/basic_vae

python vae.py --hidden_dim 128 --latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12
```

To use as template for research (example of modifying only the prior)
```python
from pytorch_lightning_bolts.models.vaes import VAE

class MyVAEFlavor(VAE):

    def get_posterior(self, mu, std):
        # do something other than the default
        # P = self.get_distribution(self.prior, loc=torch.zeros_like(mu), scale=torch.ones_like(std))

        return P
```

Or pass in your own encoders and decoders

```python
from pytorch_lightning_bolts.models.vaes import VAE
import pytorch_lightning as pl

encoder = MyEncoder()
decoder = MyDecoder()

vae = VAE(encoder=encoder, decoder=decoder)
trainer = pl.Trainer(gpus=1)
trainer.fit(vae)
```
