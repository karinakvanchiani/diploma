import torch
import torch.nn as nn, n
import torch.nn.functional as f
import torch.optim as optim


class Training():
    def __init__(self):
        super().__init__()
        self.disc_losss = n.BCEWithLogitsLoss()
        self.gen_losss = n.BCEWithLogitsLoss()
        self.vgg_loss = n.MSELoss()
        self.mse_loss = n.MSELoss()
        self.lamda = 0.005
        self.eeta = 0.02 
        
    def train(self, discriminator, generator, LR_image, HR_image, disc_optimizer, gen_optimizer, vgg, epoch):

        disc_optimizer.zero_grad()
        generated_output = generator(LR_image.to(device).float())
        fake_data = generated_output.clone()
        fake_label = discriminator(fake_data)
        
        HR_image_tensor = HR_image.to(device).float()
        real_data = HR_image_tensor.clone()
        real_label = discriminator(real_data)

        relativistic_d1_loss = self.disc_losss((real_label - torch.mean(fake_label)), torch.ones_like(real_label, dtype = torch.float))
        relativistic_d2_loss = self.disc_losss((fake_label - torch.mean(real_label)), torch.zeros_like(fake_label, dtype = torch.float))      

        d_loss = (relativistic_d1_loss + relativistic_d2_loss) / 2
        d_loss.backward(retain_graph = True)
        disc_optimizer.step()

        fake_label_ = discriminator(generated_output)
        real_label_ = discriminator(real_data)
        gen_optimizer.zero_grad()

        g_real_loss = self.gen_losss((fake_label_ - torch.mean(real_label_)), torch.ones_like(fake_label_, dtype = torch.float))
        g_fake_loss = self.gen_losss((real_label_ - torch.mean(fake_label_)), torch.zeros_like(fake_label_, dtype = torch.float))
        g_loss = (g_real_loss + g_fake_loss) / 2
        
        v_loss = self.vgg_loss(vgg.features[:6](generated_output), vgg.features[:6](real_data))
        m_loss = self.mse_loss(generated_output, real_data)
        generator_loss = self.lamda * g_loss + v_loss + self.eeta * m_loss
        generator_loss.backward()
        gen_optimizer.step()

        return d_loss, generator_loss, fake_data


def trained_gan(HR_images, LR_images, batch_size=8, epochs=10, lr_gan=0.0001, lr_disc=0.0001):
    
    data_tr = DataLoader(list(zip(HR_images[np.newaxis], LR_images[np.newaxis])), batch_size=batch_size)
    
    vgg = models.vgg19(pretrained=True).to(device)
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    gen_optimizer = optim.Adam(gen.parameters(), lr=lr_gan)
    disc_optimizer = optim.Adam(disc.parameters(), lr=lr_disc)

    generated_images = []

    for epoch in (range(epochs)):
        dloss_list, gloss_list = [], []

        for HR_data, LR_data in data_tr:
            for i in range(HR_data.shape[0]):
                torch.cuda.empty_cache()
                HR, LR = HR_data[i].permute(0, 3, 1, 2), LR_data[i].permute(0, 3, 1, 2)
                disc_loss, gen_loss, gen_image = Training().train(disc, gen, LR, HR, disc_optimizer, 
                                                                  gen_optimizer, vgg, epoch)
                dloss_list.append(disc_loss.item())
                gloss_list.append(gen_loss.item())
                torch.cuda.empty_cache()

        print('epoch: ', epoch + 1, '  ', 'd_loss: ', round(np.mean(dloss_list), 3), '  ', 'g_loss: ', round(np.mean(gloss_list), 3))
        #torch.save(gen.state_dict(), f'/content/gdrive/My Drive/ВКР/ВКР (4 курс)/datasets/test_models/gan/{epoch + 1}.torch')

    return gen, disc 


def test(HR, LR, gen, batch_size=8):

    HR, LR = torch.from_numpy(HR[np.newaxis]).permute(0, 3, 1, 2), torch.from_numpy(LR[np.newaxis]).permute(0, 3, 1, 2)
    generated_output = gen(LR.to(device).float())
    fake_data = generated_output.clone()

    HR_image_tensor = HR.to(device).float()
    real_data = HR_image_tensor.clone()

    HR_ground_truth = real_data.permute(0, 2, 3, 1)[0, :, :, :].cpu().detach().numpy()
    LR_images = LR.permute(0, 2, 3, 1)[0, :, :, :].cpu().detach().numpy()
    HR_generated = fake_data.permute(0, 2, 3, 1)[0, :, :, :].cpu().detach().numpy()

    return HR_ground_truth, LR_images, HR_generated