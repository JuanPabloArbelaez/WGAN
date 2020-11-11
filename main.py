from wgan import *
from visualize import *


## Training Initializations
N_EPOCHS = 100  
Z_DIM = 64
DISPLAY_STEP = 50
BATCH_SIZE = 128
LR = 0.0002
BETA_1 = 0.5
BETA_2 = 0.999
C_LAMBDA = 10
CRIT_REPEATS = 5
DEVICE = 'cuda'


# Load DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True)

# Initialize Generator, Critic & optimizers
gen = Generator(Z_DIM).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LR, betas=(BETA_1, BETA_2))
crit = Critic().to(DEVICE)
crit_opt = torch.optim.Adam(crit.parameters(), lr=LR, betas=(BETA_1, BETA_2))

# Weights Init
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)


def run_neural_network():
    cur_step = 0
    generator_losses = []
    critic_losses = []
    for epoch in range(N_EPOCHS):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(DEVICE)

            mean_iteration_critic_loss = 0
            for _ in range(CRIT_REPEATS):
                ## Update critic
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, Z_DIM, DEVICE)
                fake = gen(fake_noise).detach()
                fake_crit_pred = crit(fake)
                real_crit_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=DEVICE, requires_grad=True)
                gradient = get_gradient(crit, real, fake, epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(fake_crit_pred, real_crit_pred, gp, C_LAMBDA)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / CRIT_REPEATS

                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]

            ## Update Generator
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, Z_DIM, DEVICE)
            fake_2 = gen(fake_noise_2)
            fake_crit_pred = crit(fake_2)

            gen_loss = get_gen_loss(fake_crit_pred)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            # Keep track of average generator loss
            generator_losses += [gen_loss.item()]

            ## Visualization code
            if (cur_step % display_step == 0) and (cur_step > 0):
                gen_mean = sum(generator_losses[-DISPLAY_STEP:] / DISPLAY_STEP)
                crit_mean = sum(critic_losses[-DISPLAY_STEP:] / DISPLAY_STEP)
                print(f"Epoch: {epoch}  Generator loss: {gen_mean}  Critic loss: {crit_loss}")
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss")
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(crit_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss")
                plt.legend()
                plt.show()
            
            cur_step += 1