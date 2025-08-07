# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rcdefaults()

#plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# %matplotlib inline

# %%
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

#fig.tight_layout()

# %%
# --- 1. Load and Preprocess MNIST Data ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis] # Shape: (60000, 28, 28, 1)

x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis] # Shape: (10000, 28, 28, 1)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# rng = np.random.default_rng()

# model = keras.Sequential(
#     [
#         keras.layers.Input(shape=(28, 28, 1)),
#         Conv2D(32, 5, activation='relu'),
#         Conv2D(64, 5, activation='relu'),
#         Conv2DTranspose(64, 5, activation='relu'),
#         Conv2DTranspose(32, 5, activation='relu'),
#         Conv2DTranspose(1, 1, activation='sigmoid')
#     ]
# )

# model.summary()


# %%
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=losses.MeanSquaredError()
)

BATCH_SIZE = 128
EPOCHS = 10

history = autoencoder.fit(
    x_train, x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=(x_test, x_test)
)



# %%
autoencoder.encoder.summary(), autoencoder.decoder.summary() 

# %%
# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (Normal Data)')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()


# --- 4. Detect "Weird-Looking" Digits in x_test ---
# Reconstruct all images in x_test
reconstructed_x_test = autoencoder.predict(x_test)

# Calculate Mean Squared Error (MSE) for each image in x_test
mse_per_test_image = np.mean(np.square(x_test - reconstructed_x_test), axis=(1, 2, 3))

# --- 5. Determine a Threshold for "Weirdness" ---
# The threshold is now based on the distribution of reconstruction errors on x_test itself.
# We are looking for outliers in this distribution.
# A common way is to use a percentile.
# For example, flag the top 1% or 5% with the highest reconstruction error as "weird".
WEIRDNESS_PERCENTILE = 99.9 # Flag top 1% as weird
threshold = np.percentile(mse_per_test_image, WEIRDNESS_PERCENTILE)

print(f"\nReconstruction MSE distribution on x_test (all digits):")
print(f"  Mean: {np.mean(mse_per_test_image):.6f}")
print(f"  Std Dev: {np.std(mse_per_test_image):.6f}")
print(f"  Median: {np.median(mse_per_test_image):.6f}")
print(f"  {WEIRDNESS_PERCENTILE}th Percentile: {threshold:.6f}")
print(f"Chosen 'Weirdness' Threshold (MSE based on {WEIRDNESS_PERCENTILE}th percentile of x_test errors): {threshold:.6f}")

# Identify "weird" images in the x_test set
weird_indices = np.where(mse_per_test_image > threshold)[0]
true_labels_of_weird = y_test[weird_indices] # Get the actual digits for these

print(f"\nOut of {len(x_test)} test samples:")
print(f"  {len(weird_indices)} were flagged as 'weird-looking' (MSE > threshold).")
if len(weird_indices) > 0:
    print(f"  Actual labels of these 'weird' digits: {true_labels_of_weird[:20]} (showing first 20 if many)")


# --- 6. Plot Samples with High Reconstruction Error from x_test ---
num_samples_to_plot = 10
# Get indices of images with highest MSE, these are our "weird" candidates
indices_sorted_by_mse_desc = np.argsort(mse_per_test_image)[::-1]

# Select the top 'num_samples_to_plot' from these, which should all be above the threshold
# if WEIRDNESS_PERCENTILE is chosen appropriately (e.g., not 0 or 100).
# If you want to guarantee they are above threshold:
# plot_indices = [idx for idx in indices_sorted_by_mse_desc if mse_per_test_image[idx] > threshold][:num_samples_to_plot]
# Simpler for now, just take the top N by MSE:
plot_indices = indices_sorted_by_mse_desc[:num_samples_to_plot]


plotted_count = 0
print(f"\nPlotting {num_samples_to_plot} samples from x_test with the highest reconstruction error (potential weird digits):")

fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(6, 2 * num_samples_to_plot + 1)) # +1 for suptitle room
fig.suptitle(f"x_test Samples with Highest Reconstruction Error\n(Above {WEIRDNESS_PERCENTILE}th percentile MSE)", fontsize=14)

for i, original_idx in enumerate(plot_indices):
    if plotted_count >= num_samples_to_plot:
        break

    original_img = x_test[original_idx].squeeze()
    reconstructed_img = reconstructed_x_test[original_idx].squeeze()
    current_mse = mse_per_test_image[original_idx]
    true_label = y_test[original_idx]


    # Plot original
    ax = axes[plotted_count, 0]
    ax.imshow(original_img, cmap='gray')
    ax.set_title(f"Original (Digit {true_label})", fontsize=8)
    ax.axis('off')

    # Plot reconstructed
    ax = axes[plotted_count, 1]
    ax.imshow(reconstructed_img, cmap='gray')
    ax.set_title(f"Reconstructed\nMSE: {current_mse:.4f}", fontsize=8)
    ax.axis('off')

    plotted_count += 1

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle
plt.show()

# --- Optional: Plot some well-reconstructed samples from x_test ---
num_normal_samples_to_plot = 5
indices_sorted_by_mse_asc = np.argsort(mse_per_test_image) # Sort by MSE ascending
plot_indices_normal = indices_sorted_by_mse_asc[:num_normal_samples_to_plot]

fig_normal, axes_normal = plt.subplots(num_normal_samples_to_plot, 2, figsize=(6, 2*num_normal_samples_to_plot + 1))
fig_normal.suptitle(f"x_test Samples with Lowest Reconstruction Error (Well Reconstructed)", fontsize=14)

for i, original_idx in enumerate(plot_indices_normal):
    original_img = x_test[original_idx].squeeze()
    reconstructed_img = reconstructed_x_test[original_idx].squeeze()
    current_mse = mse_per_test_image[original_idx]
    true_label = y_test[original_idx]

    ax = axes_normal[i, 0]
    ax.imshow(original_img, cmap='gray')
    ax.set_title(f"Original (Digit {true_label})", fontsize=8)
    ax.axis('off')

    ax = axes_normal[i, 1]
    ax.imshow(reconstructed_img, cmap='gray')
    ax.set_title(f"Reconstructed\nMSE: {current_mse:.4f}", fontsize=8)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
rng = np.random.default_rng(seed=13)

index = rng.choice(np.arange(len(x_test)))

image = x_test[index]
rec_image = reconstructed_x_test[index]

fig,axes = plt.subplots(ncols=2, nrows=1, figsize=(6,4))

axes[0].imshow(image)
axes[1].imshow(rec_image)

for ax in axes:
    ax.axis('off')

fig.tight_layout()
fig.savefig('mnist_auto.png')

# %%
index_anom = rng.choice(np.arange(len(plot_indices)))

image = x_test[index_anom]
rec_image = reconstructed_x_test[index_anom]

fig,axes = plt.subplots(ncols=2, nrows=1, figsize=(6,4))

axes[0].imshow(image)
axes[1].imshow(rec_image)

for ax in axes:
    ax.axis('off')

fig.tight_layout()
fig.savefig('mnist_auto_anom.png')

# %%
from tensorflow.keras.utils import plot_model

plot_model(
    autoencoder.encoder, to_file='./model_plot.png', 
    show_shapes=True, show_layer_names=True, show_dtype=False, 
    rankdir='TB'
)

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the range for u and v
u_min, u_max = -2.5, 2.5 # Adjusted range to better see the cubic features
v_min, v_max = -2.5, 2.5
num_points = 200  # Increased points for smoother contours

u_vals = np.linspace(u_min, u_max, num_points)
v_vals = np.linspace(v_min, v_max, num_points)

# 2. Create a 2D grid (meshgrid) of u and v values
U, V = np.meshgrid(u_vals, v_vals)

# 3. Calculate Re(x^3) and Im(x^3)
Re_X_cubed = U**3 - 3*U*(V**2)
Im_X_cubed = 3*(U**2)*V - V**3

# 4. Plot the contours
plt.figure(figsize=(9, 7)) # Adjusted figure size

# Determine suitable contour levels.
# For cubic functions, values can grow quickly.
# Let's try some levels around zero and some larger ones.
# You might need to adjust these based on the u,v range.
levels_re = np.array([-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8])
levels_im = np.array([-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8])
# Alternatively, you can let matplotlib choose, or use np.linspace for evenly spaced levels
# levels_re = plt.MaxNLocator(nbins=15).tick_values(Re_X_cubed.min(), Re_X_cubed.max())
# levels_im = plt.MaxNLocator(nbins=15).tick_values(Im_X_cubed.min(), Im_X_cubed.max())


# Plot contours of constant Re(x^3)
cs_re = plt.contour(U, V, Re_X_cubed, levels=levels_re, colors='blue', linestyles='solid', linewidths=1.5)
plt.clabel(cs_re, inline=True, fontsize=9, fmt='%1.1f') # Add labels to contours

# Plot contours of constant Im(x^3)
cs_im = plt.contour(U, V, Im_X_cubed, levels=levels_im, colors='red', linestyles='dashed', linewidths=1.5)
plt.clabel(cs_im, inline=True, fontsize=9, fmt='%1.1f') # Add labels to contours

# 5. Add labels and title
plt.xlabel('u (Real part of x)')
plt.ylabel('v (Imaginary part of x)')
plt.title('Contours of Re(x³) and Im(x³) in the u,v plane (x = u + iv)')

# Add x and y axes for reference
plt.axhline(0, color='black', lw=0.7)
plt.axvline(0, color='black', lw=0.7)

# Set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)

# Create a custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='blue', lw=1.5, linestyle='solid', label='Re(x³) = constant'),
                   Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label='Im(x³) = constant')]
plt.legend(handles=legend_elements, loc='upper right')

plt.show()

# %% [markdown]
# $$
# \int\limits_0^1{\rm d}x\,e^{i\lambda x^3}
# $$
# $$
# x = u+iv\,, \qquad \Re x^3 = {\rm const}\,, \quad \Im x^3 = {\rm const}\,. 
# $$

# %%
