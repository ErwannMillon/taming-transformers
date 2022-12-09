from img_processing import custom_to_pil, preprocess, preprocess_vqgan
import matplotlib.pyplot as plt
import PIL
def get_latent_from_path(path, model):
	DEVICE = "mps"
	x = preprocess(PIL.Image.open(path), target_image_size=256, map_dalle=False)
	
	x = x.to(DEVICE)

	x_processed = preprocess_vqgan(x)
	x_latent, _, [_, _, indices] = model.encode(x_processed)
	return x_latent
	
    
def show_latent(model, latent):

    dec = model.decode(latent.to("mps"))
    im = custom_to_pil(dec[0])
    plt.figure(figsize=(3, 3))
    plt.imshow(im)
    return im