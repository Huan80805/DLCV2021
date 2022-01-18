import torch
import torch.optim as optim
import os
import sys
import wandb
from torchvision.utils import save_image

def gradient_penalty(netD, real_data, fake_data):
  batch_size = real_data.size(0)
  # Sample Epsilon from uniform distribution
  eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
  eps = eps.expand_as(real_data)
  
  # Interpolation between real data and fake data.
  interpolation = eps * real_data + (1 - eps) * fake_data
  
  # get logits for interpolated images
  interp_logits, _ = netD(interpolation)
  grad_outputs = torch.ones(interp_logits.size()).to(real_data.device)
  
  # Compute Gradients
  gradients = torch.autograd.grad(
    outputs=interp_logits,
    inputs=interpolation,
    grad_outputs=grad_outputs,
    create_graph=True,
    retain_graph=True,
    only_inputs=True
  )[0]
  
  # Compute and return Gradient Norm
  gradients = gradients.view(batch_size, -1)
  grad_norm = gradients.norm(2, 1)
  return torch.mean((grad_norm - 1) ** 2)

def train(config, dataloader, G, D):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  # Loss functions
  TF_criterion = torch.nn.BCELoss()
  class_criterion = torch.nn.CrossEntropyLoss()
  # Optimizers
  optim_G = optim.Adam(G.parameters(), lr=config.lr_G, betas=(0.5, 0.999))
  optim_D = optim.Adam(D.parameters(), lr=config.lr_D, betas=(0.5, 0.999))

  if config.lr_scheduler == "step":
    optim_G_scheduler = optim.lr_scheduler.StepLR(optim_G, config.lr_decay_epoch, 0.5)
    optim_D_scheduler = optim.lr_scheduler.StepLR(optim_D, config.lr_decay_epoch, 0.5)
  else:
    optim_G_scheduler = None
    optim_D_scheduler = None
  
  def label_smoothing(batch_size, smoothed):
    if smoothed:
        #real: 0.7-1.2, fake:0-0.3
        real_label = torch.rand(batch_size,)*0.5 + 0.7
        fake_label = torch.rand(batch_size,)*0.3
    else:
        real_label = torch.full((batch_size,),1,dtype=torch.float).to(device)
        fake_label = torch.full((batch_size,),0,dtype=torch.float).to(device)
    return real_label, fake_label



  iteration = 0
  real_acc_total = 0
  sample_acc_total = 0
  D_realloss_total = 0
  D_fakeloss_total = 0
  G_loss_total = 0
  fixed_z = torch.randn(10,100).to(device)
  for ep in range(config.epochs):
    for batch_idx, (real_data, real_class) in enumerate(dataloader):
      real_data, real_class = real_data.to(device), real_class.to(device)                        
      batch_size = real_data.size(0)
      ##gounrdtruth
      real_label, fake_label = label_smoothing(batch_size,config.smoothed)
      real_label, fake_label = real_label.to(device), fake_label.to(device)
      ##Generator input
      sample_class = torch.randint(0,10,(batch_size,)).to(device)
      z = torch.randn(batch_size,100).to(device)
      ##Generate images
      gen_imgs = G(z, sample_class)
      #-------TRAINING DISCRIMINATOR-----
      optim_D.zero_grad()      
      real_pred_class, real_pred_label = D(real_data)
      fake_pred_class, fake_pred_label = D(gen_imgs)
      if config.model_name == "WGAN":
        # TF:real-to-fake distance
        D_realloss = (-torch.mean(real_pred_label) + config.class_weight*class_criterion(real_pred_class,real_class))
        D_fakeloss = (torch.mean(fake_pred_label) + config.class_weight*class_criterion(fake_pred_class, sample_class))
        D_loss = (D_realloss + D_fakeloss + 10 * gradient_penalty(D, real_data, gen_imgs)) / 2
      else:
        sys.exit("WGAN available only")
      D_loss.backward()
      optim_D.step()        
      ## Statistic
      real_acc_total += torch.mean((torch.argmax(real_pred_class, dim=1) == real_class).to(torch.float))
      sample_acc_total += torch.mean((torch.argmax(fake_pred_class, dim=1) == sample_class).to(torch.float))
      D_realloss_total += D_realloss
      D_fakeloss_total += D_fakeloss
      if iteration % config.n_critic == 0:
      #------Training Generator------
        ##onlt training generator on every n_critic
        optim_G.zero_grad()
        ##Generator input
        sample_class = torch.randint(0,10,(batch_size,)).to(device)
        z = torch.randn(batch_size,100).to(device)
        # Generate images
        gen_imgs = G(z, sample_class)
        # Loss measures generator's ability to fool the discriminator
        fake_pred_class, fake_pred_label = D(gen_imgs)
        G_loss = 0.5*(-torch.mean(fake_pred_label) + config.class_weight*class_criterion(fake_pred_class, sample_class))
        G_loss.backward()
        optim_G.step()
      #statistic
      G_loss_total += G_loss
      if iteration % config.log_interval == 0 and iteration > 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)] D_loss: [real:{:.6f}, fake:{:.6f}]'.format(
        ep, batch_idx * len(real_data), 
        len(dataloader.dataset),
        100. * batch_idx / len(dataloader),
        D_realloss_total.item()/config.log_interval,
        D_fakeloss_total.item()/config.log_interval), end='')
        print(' G_LOSS: {:.6f}'.format(
            G_loss_total.item()/(config.log_interval/config.n_critic)),end='')
        print(' ACC: [sample:{:.6f}%, real:{:.6f}%]'.format(
            100.*sample_acc_total.item()/config.log_interval,
            100.*real_acc_total.item()/config.log_interval))
        wandb.log({
        "G_LOSS":G_loss_total.item()/config.log_interval,
        "Sample_ACC":sample_acc_total.item()/config.log_interval,
        "Real_ACC":real_acc_total.item()/config.log_interval,                
        "D_FAKELOSS":D_fakeloss_total.item()/config.log_interval,
        "D_REALLOSS":D_realloss_total.item()/config.log_interval                               
        })
        real_acc_total = 0
        sample_acc_total = 0
        D_realloss_total = 0
        D_fakeloss_total = 0
        G_loss_total = 0
      iteration += 1
    test(G,ep,fixed_z,config)
    if ep %5 ==0:
        save_checkpoint(os.path.join(config.ckpt_path,'G_epoch{}.pth'.format(ep)), G, optim_G)
        save_checkpoint(os.path.join(config.ckpt_path,'D_epoch{}.pth'.format(ep)), D, optim_D)
    if optim_D_scheduler != None:
        optim_D_scheduler.step()
    if optim_G_scheduler != None:
      optim_G_scheduler.step()
  # save the final model
  save_checkpoint(os.path.join(config.ckpt_path,'G_epoch{}.pth'.format(ep)), G, optim_G)
  save_checkpoint(os.path.join(config.ckpt_path,'D_epoch{}.pth'.format(ep)), D, optim_D)

def test(G,ep,z,config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.eval()
    test_c = torch.arange(0,10,1).to(device)
    test_z = z
    image = G(test_z,test_c)
    image = ((image+1)/2).clamp(0,1)
    save_image(image, os.path.join(config.test_folder,str(ep)+".png"),nrow=5)
        

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
        

