import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from ..models.diffusion import Model
from ..models.ema import EMAHelper
from ..functions import get_optimizer
from ..functions.losses import loss_registry
from ..datasets import get_dataset, data_transform, inverse_data_transform
from ..functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                if config.data.random_flip:
                    if torch.randint(0, 2, (1,))[0].item() == 0 :
                        x = torch.flip(x, dims=[-1])
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}, epoch: {epoch}"
                )


                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    # torch.save(
                    #     states,
                    #     os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    # )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            
            states = torch.load(
                self.args.ckpt_path,
                map_location=self.config.device)
                
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states["ddim_state_dict"], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states["ddim_state_dict_ema"])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10" or self.config.data.dataset == 'CIFAR100':
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        elif self.args.dataset:
            self.sample_dataset(model)
        elif self.args.sample_latent:
            self.sample_latent(model)
        elif self.args.rec_from_latent:
            self.rec_from_latent(model)
        elif self.args.autoencoder:
            self.sample_autoencoder(model)
        elif self.args.reconstruct:
            self.reconstruct(model)
        elif self.args.pca_interpolation:
            self.pca_interpolation(model)
        elif self.args.pca_interpolation_moco_latent:
            self.pca_interpolation_moco_latent(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        args = self.args
        print(f"starting sampling")
        total_n_samples = args.n_samples
        n_rounds = (total_n_samples) // config.sampling.batch_size

        samples = []
        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)

                samples.extend(x.cpu().data.numpy())
        samples = np.array(samples)
        print(samples.shape)
        out_fname = os.path.join(self.args.log_path, str(args.timesteps) + "_samples.npy")
        # out_fname = "samples/" + config.data.dataset + "_" + args.doc + "_samples.npy"
        np.save(out_fname, samples)

    def sample_autoencoder(self, model):
        config = self.config
        args = self.args
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        x = next(iter(train_loader))[0]
        x = x.to(self.device)
        x = data_transform(self.config, x)

        from functions.denoising import generalized_steps, encoding_steps

        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        y = encoding_steps(x, seq, model, self.betas)[-1].to("cuda")
        print(y.mean().item(), y.std().item())
        xs = generalized_steps(y, seq, model, self.betas, eta=self.args.eta)
        x_ = xs[0][-1].to("cuda")

        loss = (x - x_).square().sum(dim=(1, 2, 3)).mean(dim=0)
        print(loss.item())

    def sample_dataset(self, model):
        config = self.config
        total_n_samples = 50000
        img_id = 0
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        dataset_x = []
        dataset_y = []

        dataset, test_dataset = get_dataset(self.args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        with torch.no_grad():
            for y, _ in tqdm.tqdm(train_loader, desc="Generating dataset"):
                seq = range(0, self.num_timesteps)
                y = y.to('cuda')

                from functions.denoising import encoding_steps
                x = encoding_steps(y, seq, model, self.betas)[-1].to("cuda")

                dataset_x.append(x.to("cpu"))
                dataset_y.append(y.to("cpu"))
        dataset_x = torch.cat(dataset_x, dim=0)
        dataset_y = torch.cat(dataset_y, dim=0)

        torch.save(
            {"x": dataset_x, "y": dataset_y},
            os.path.join(self.args.image_folder, f"dataset.pth"),
        )

    def get_latent(self, loader, model, fname):
        config = self.config
        args = self.args

        feats = []
        labels = []

        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                print(x.size())
                x = x.to(self.device)
                # x = data_transform(self.config, x)

                from functions.denoising import generalized_steps, encoding_steps

                if self.args.skip_type == "uniform":
                    skip = self.num_timesteps // self.args.timesteps
                    seq = range(0, self.num_timesteps, skip)
                elif self.args.skip_type == "quad":
                    seq = (
                        np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                        ** 2
                    )
                    seq = [int(s) for s in list(seq)]
                else:
                    raise NotImplementedError

                print(i, len(loader))

                latent = encoding_steps(x, seq, model, self.betas)[-1].to("cuda")
                feats.extend(latent.cpu().data.numpy())
                # labels.extend(y.cpu().data.numpy())
                if len(feats) >= 25000:
                    break


            feats = np.array(feats)[:25000]
            feats = np.reshape(feats, [len(feats), -1])
            # labels = np.array(labels)[:10000]

            print(feats.shape)
            # print(labels.shape)
            np.save(fname, feats)
            # np.save(fname.replace(".npy", "_labels.npy"), labels)

    def sample_latent(self, model):
        config = self.config
        args = self.args
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        # self.get_latent(train_loader, model, "feats_orig/C10_train_" + str(args.timesteps) + ".npy")
        self.get_latent(test_loader, model, "feats_our/CA_test_" + str(args.timesteps) + ".npy")
    
    def reconstruct(self, model):
        config = self.config
        args = self.args

        latents = np.load(args.latent_fname)
        print("inside ddim and the size of latents is ", latents.shape)

        data_mean = np.load(args.mean_file)
        data_std = np.load(args.std_file)

        latents = (latents - data_mean)/data_std
        latents = torch.from_numpy(latents).float()

        x = latents.to("cuda")
        
        if args.noise_add_steps == 0:
            from functions.denoising import generalized_steps, encoding_steps

            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            print(seq)
            y = encoding_steps(x, seq, model, self.betas)[-1].to("cuda")
            xs = generalized_steps(y, seq, model, self.betas, eta=self.args.eta)
            x_ = xs[0][-1].to("cuda")
        else:
            print('add noise then denoise mode')
            with torch.no_grad():
                from functions.denoising import generalized_steps, compute_alpha
                a = compute_alpha(self.betas,
                                  args.noise_add_steps * torch.ones(x.size(0)).long().to(x.device)).to('cuda')
                x = a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x)
                seq = list(range(0, args.noise_add_steps, args.noise_skip_steps)) + [args.noise_add_steps]
                x_ = generalized_steps(x, seq, model, self.betas, eta=args.eta)[0][-1]

        out_fname = args.out_fname
        print("the out fname is ", out_fname)
        x_ = x_.cpu().data.numpy()
        x_ = x_*data_std + data_mean
        print(x_.shape)
        np.save(out_fname, x_)

    def pca_interpolation(self, model):
        config = self.config
        args = self.args
        np.random.seed(int(time.time()))
        batch_size = 5
        resolution = 64
        interps = 10
        pca_component_idx = 0

        # latents = np.load("feats_orig/C10_train_100.npy")
        latents = np.load("feats_our/CA_test_100.npy")
        np.random.shuffle(latents)


        latents = latents[:batch_size]
        latents = np.repeat(latents, interps, axis=0)
        interp_strength = np.reshape(np.array([2.0*i for i in range(interps)]), [-1, 1])
        interp_strength = np.tile(interp_strength, [batch_size, 1])
        print(interp_strength)

        pca_components = np.load("pca/CA_pca_components_our_25k.npy")
        pca_component = pca_components[pca_component_idx]

        interp_directions = np.reshape(pca_component, [1, -1])*interp_strength

        latents_new = latents + interp_directions
        latents_new = torch.from_numpy(latents_new).float()
        print("the size is ", latents_new.size())
        latents_new = latents_new.to(self.device)


        from functions.denoising import generalized_steps, encoding_steps

        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError


        x = latents_new.view((-1, 5, 32, 32))
        xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
        x_ = xs[0][-1].to("cuda")

        out_fname = os.path.join(self.args.log_path, str(args.timesteps) + "_pca_interp.npy")
        print(x_.cpu().data.numpy().shape)
        np.save(out_fname, x_.cpu().data.numpy())

    def pca_interpolation_moco_latent(self, model):
        config = self.config
        args = self.args
        np.random.seed(int(time.time()))
        batch_size = 5
        resolution = 64
        interps = 10
        pca_component_idx = 9

        # latents = np.load("feats_orig/C10_train_100.npy")
        latents = np.load("../NVAE/feats_cls_celeba/train_feats_ae_moco_32_rot.npy")
        np.random.shuffle(latents)


        latents = latents[:batch_size]
        latents = np.reshape(latents, [len(latents), -1])
        latents = np.repeat(latents, interps, axis=0)
        interp_strength = np.reshape(np.array([25.0*i for i in range(interps)]), [-1, 1])
        interp_strength = np.tile(interp_strength, [batch_size, 1])
        print(interp_strength)

        pca_components = np.load("pca/CA_pca_components_our_moco_25k.npy")
        pca_component = pca_components[pca_component_idx]

        interp_directions = np.reshape(pca_component, [1, -1])*interp_strength

        latents_new = latents + interp_directions
        latents_new = torch.from_numpy(latents_new).float()
        print("the size is ", latents_new.size())
        latents_new = latents_new.to(self.device)



        x = latents_new.view((-1, 5, 32, 32))

        out_fname = os.path.join(self.args.log_path, str(args.timesteps) + "_pca_interp_moco.npy")
        print(x.cpu().data.numpy().shape)
        np.save(out_fname, x.cpu().data.numpy())

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            16,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        new_x = torch.stack(x[::50], axis=0)

        new_x = new_x.cpu().data.numpy()
        print(new_x.shape)
        out_fname = os.path.join(self.args.log_path, str(self.args.timesteps) + "_intermediate_samples.npy")
        # out_fname = "samples/" + config.data.dataset + "_" + args.doc + "_samples.npy"
        np.save(out_fname, new_x)

    def rec_from_latent(self, model):
        config = self.config
        args = self.args
        data_mean = np.load(args.mean_file)
        data_std = np.load(args.std_file)
        latents = np.load(args.inp_fname)
        latents = torch.from_numpy(latents).float()

        from functions.denoising import generalized_steps, encoding_steps

        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps)
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        labels = torch.zeros(latents.size(0)).float()
        dataset = torch.utils.data.TensorDataset(
            latents, labels)
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=50,
                shuffle=False,
                num_workers=8,
            )
        x_all= []
        # data_mean = np.load("/atlas/u/a7b23/winter_2021/NVAE/out_all/FFHQ_attributes_32/train_feats_ffhq_32_mean.npy")
        # data_std = np.load("/atlas/u/a7b23/winter_2021/NVAE/out_all/FFHQ_attributes_32/train_feats_ffhq_32_std.npy")

        data_mean = np.load(args.mean_file)
        data_std = np.load(args.std_file)
        
        out_fname = args.out_fname
        from functions.denoising import generalized_steps, compute_alpha

        if args.noise_add_steps == 0:
            with torch.no_grad():
                for i, (x, _) in enumerate(dataloader):
                    print(i, len(dataloader), x.size())
                    x = x.to("cuda")
                    xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
                    x_ = xs[0][-1].to("cuda")
                    x_ = x_.cpu().data.numpy()
                    x_ = x_*data_std + data_mean
                    x_all.extend(x_)
                    del xs
                del x
        else:
            print('add noise then denoise')
            with torch.no_grad():
                for i, (x, _) in enumerate(dataloader):
                    print(i, len(dataloader), x.size())
                    x = (x - torch.from_numpy(data_mean)) / torch.from_numpy(data_std)
                    x = x.to('cuda')
                    a = compute_alpha(self.betas.to('cuda'),
                                    args.noise_add_steps * torch.ones(x.size(0)).long().to(x.device)).to('cuda')
                    x = a.sqrt() * x + (1 - a).sqrt() * torch.randn_like(x)
                    seq = list(range(0, args.noise_add_steps, args.noise_skip_steps)) + [args.noise_add_steps]
                    x_ = generalized_steps(x, seq, model, self.betas, eta=args.eta)[0][-1]
                    x_ = x_.cpu().data.numpy()
                    x_ = x_*data_std + data_mean
                    x_all.extend(x_)

        x_all  = np.array(x_all)
        print(x_all.shape)
        np.save(out_fname, x_all)

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            # for i in range(0, x.size(0), 8):
            #     xs.append(self.sample_image(x[i : i + 8], model).cpu().data.numpy())
            #     print(xs[-1].shape)
            xs_tensor = self.sample_image(x, model)

        xs = np.array(xs_tensor.cpu().data.numpy())
        print(xs.shape)

        out_fname = os.path.join(self.args.log_path, str(self.args.timesteps) + "_interp_latent.npy")
        # out_fname = "samples/" + config.data.dataset + "_" + args.doc + "_samples.npy"
        np.save(out_fname, xs)

        latent1 = xs_tensor[0:1].to(self.device)
        latent2 = xs_tensor[-1:].to(self.device)

        latents = []

        for i in range(alpha.size(0)):
            latents.append(slerp(latent1, latent2, alpha[i]))

        latents = torch.cat(latents, dim=0)
        latents = latents.cpu().data.numpy()
        print(latents.shape)
        out_fname = os.path.join(self.args.log_path, str(self.args.timesteps) + "_interp_latent_orig.npy")
        np.save(out_fname, latents)



    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
