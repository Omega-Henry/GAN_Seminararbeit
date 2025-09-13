from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained generator
generator = load_model("generator_mnist.h5")

# Generate new digits
noise = np.random.normal(0, 1, (25, 100))  # 100 = latent dim
gen_imgs = generator.predict(noise)

# Rescale 0–1
gen_imgs = 0.5 * gen_imgs + 0.5

# Plot
r, c = 5, 5
fig, axs = plt.subplots(r, c, figsize=(6, 6))
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
        axs[i, j].axis("off")
        cnt += 1
plt.tight_layout()
plt.savefig("realistic_fake_digits.png")
plt.close()
print("Saved image of realistic fake digits.")
