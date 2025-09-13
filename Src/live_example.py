import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load your trained generator
generator = load_model('generator.h5')

# Generate random noise
noise = np.random.normal(0, 1, (10, 100))

# Generate images
gen_imgs = generator.predict(noise)

# Rescale from [-1, 1] to [0, 1]
gen_imgs = 0.5 * gen_imgs + 0.5

# Plot the results
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(gen_imgs[i, :, :, 0], cmap='gray')
    plt.axis('off')

plt.suptitle("GAN-Generated Digits (Live Example)")
plt.tight_layout()
plt.show()
