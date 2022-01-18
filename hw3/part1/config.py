class Config(object):
    def __init__(self, exp_name):
        self.model_name = 'VIT_B16'
        self.ckpt_path = 'best.pth'
        self.img_size = 384
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.brightness = 0.0
        self.contrast = 0.0
        self.dropout_rate = 0.1
        self.attention_dropout_rate = 0.0
        self.num_classes = 37
        self.pretrained = False
        if exp_name == 'best':
            self.ckpt_path = 'best.pth'
            self.num_heads = 12
        elif exp_name == 'best2':
            self.ckpt_path = 'best2.pth'
            self.num_heads = 12
        elif exp_name == 'best3':
            self.ckpt_path = 'best3.pth'
            self.num_heads = 8
        elif exp_name == 'best4':
            self.ckpt_path = 'best4.pth'
            self.num_heads = 16
        elif exp_name == 'best5':
            self.ckpt_path = 'best5.pth'
            self.num_heads = 12
        elif exp_name == 'best6':
            self.ckpt_path = 'best6.pth'
            self.num_heads = 12
            self.dropout_rate = 0.0
        
        

        