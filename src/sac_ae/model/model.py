import model.modules as m
import torch
import torch.nn as nn

class SAC_Model(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, device, robot_shape, cnn_stride, cnn_3dconv, robot_encoder_architecture, cost):
        super().__init__()

        shared_cnn = m.SharedCNN(obs_shape = obs_shape,
                                 num_layers = num_layers,
                                 num_filters = num_filters,
                                 stride=cnn_stride,
                                 cnn_3dconv=cnn_3dconv)

        actor_encoder = m.Encoder(cnn = shared_cnn,
                                  projection= m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))
        
        critic_encoder = m.Encoder(cnn = shared_cnn,
                                   projection = m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))

        critic_encoder_target = m.Encoder(cnn = m.SharedCNN(obs_shape = obs_shape, num_layers = num_layers, num_filters = num_filters, stride=cnn_stride, cnn_3dconv=cnn_3dconv),
                                          projection = m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))

        if robot_encoder_architecture != None:
            shared_robot_encoder = m.RobotEncoder(robot_shape, robot_encoder_architecture)
            target_robot_encoder = m.RobotEncoder(robot_shape, robot_encoder_architecture)
        else:
            shared_robot_encoder = None
            target_robot_encoder = None


        self.actor = m.Actor(encoder = actor_encoder,
                             action_dim = action_shape[0],
                             hidden_dim = hidden_dim,
                             log_std_min = log_std_min,
                             log_std_max = log_std_max,
                             robot_shape = robot_shape,
                             robot_encoder = shared_robot_encoder).to(device)
        
        self.critic = m.Critic(encoder = critic_encoder,
                                action_dim = action_shape[0],
                                hidden_dim = hidden_dim,
                                robot_shape = robot_shape,
                                robot_encoder = shared_robot_encoder).to(device)
        
        self.critic_target = m.Critic(encoder = critic_encoder_target,
                                        action_dim = action_shape[0],
                                        hidden_dim = hidden_dim,
                                        robot_shape = robot_shape,
                                        robot_encoder = target_robot_encoder).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if cost in ['critic_train', 'critic_eval']:
            cost_critic_encoder = m.Encoder(cnn = shared_cnn,
                                    projection = m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))

            cost_critic_encoder_target = m.Encoder(cnn = m.SharedCNN(obs_shape = obs_shape, num_layers = num_layers, num_filters = num_filters, stride=cnn_stride, cnn_3dconv=cnn_3dconv),
                                    projection = m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))
                                
            if robot_encoder_architecture != None:
                cost_target_robot_encoder = m.RobotEncoder(robot_shape, robot_encoder_architecture)
            else:
                cost_target_robot_encoder = None

            self.cost_critic = m.Critic(encoder = cost_critic_encoder,
                                    action_dim = action_shape[0],
                                    hidden_dim = hidden_dim,
                                    robot_shape = robot_shape,
                                    robot_encoder = shared_robot_encoder).to(device)
            
            self.cost_critic_target = m.Critic(encoder = cost_critic_encoder_target,
                                    action_dim = action_shape[0],
                                    hidden_dim = hidden_dim,
                                    robot_shape = robot_shape,
                                    robot_encoder = cost_target_robot_encoder).to(device)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())


    def soft_update_params(self, critic_tau, encoder_tau):
        for param, target_param in zip(self.critic.Q1.parameters(), self.critic_target.Q1.parameters()):
            target_param.data.copy_(critic_tau * param.data + (1 - critic_tau) * target_param.data) 
            
        for param, target_param in zip(self.critic.Q2.parameters(), self.critic_target.Q2.parameters()):
            target_param.data.copy_(critic_tau * param.data + (1 - critic_tau) * target_param.data)   

        for param, target_param in zip(self.critic.encoder.parameters(), self.critic_target.encoder.parameters()):
            target_param.data.copy_(encoder_tau * param.data + (1 - encoder_tau) * target_param.data)   
        if self.critic.robot_encoder != None:
            for param, target_param in zip(self.critic.robot_encoder.parameters(), self.critic_target.robot_encoder.parameters()):
                target_param.data.copy_(encoder_tau * param.data + (1 - encoder_tau) * target_param.data)   
        if hasattr(self, 'cost_critic'):
            for param, target_param in zip(self.cost_critic.Q1.parameters(), self.cost_critic_target.Q1.parameters()):
                target_param.data.copy_(critic_tau * param.data + (1 - critic_tau) * target_param.data) 
                
            for param, target_param in zip(self.cost_critic.Q2.parameters(), self.cost_critic_target.Q2.parameters()):
                target_param.data.copy_(critic_tau * param.data + (1 - critic_tau) * target_param.data)   

            for param, target_param in zip(self.cost_critic.encoder.parameters(), self.cost_critic_target.encoder.parameters()):
                target_param.data.copy_(encoder_tau * param.data + (1 - encoder_tau) * target_param.data)   
            if self.critic.robot_encoder != None:
                for param, target_param in zip(self.cost_critic.robot_encoder.parameters(), self.cost_critic_target.robot_encoder.parameters()):
                    target_param.data.copy_(encoder_tau * param.data + (1 - encoder_tau) * target_param.data)  
        

    
class SAC_State_Model(SAC_Model):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, log_std_max, device, cost):
        super().__init__((9, 84, 84), action_shape, hidden_dim, encoder_feature_dim, log_std_min, 
                         log_std_max, 1, 32, device, 0, 1, False, None, cost)
        actor_encoder = m.Encoder(cnn = nn.Identity(),
                                  projection= m.RLProjection(obs_shape[0], encoder_feature_dim))
        
        critic_encoder = m.Encoder(cnn = nn.Identity(),
                                   projection = m.RLProjection(obs_shape[0], encoder_feature_dim))

        critic_encoder_target = m.Encoder(cnn = nn.Identity(),
                                          projection = m.RLProjection(obs_shape[0], encoder_feature_dim))

        self.actor.encoder = actor_encoder
        self.critic.encoder = critic_encoder
        self.critic_target.encoder = critic_encoder_target
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if cost in ['critic_train', 'critic_eval']:
            cost_critic_encoder = m.Encoder(cnn = nn.Identity(),
                                    projection = m.RLProjection(obs_shape[0], encoder_feature_dim))
            cost_critic_encoder_target = m.Encoder(cnn = nn.Identity(),
                                            projection = m.RLProjection(obs_shape[0], encoder_feature_dim))
            self.cost_critic.encoder = cost_critic_encoder
            self.cost_critic_target.encoder = cost_critic_encoder_target
            self.cost_critic.to(device)
            self.cost_critic_target.to(device)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())

class CURL_Model(SAC_Model):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, 
                log_std_max, num_layers, num_filters, device):
        super().__init__(obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, 
                         log_std_max, num_layers, num_filters, device)

        self.curl = m.CURL(self.critic.encoder).to(device)


class SACAE_Model(SAC_Model):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, 
                log_std_max, num_layers, num_filters, device, robot_shape, cnn_stride, cnn_3dconv, robot_encoder_architecture, cost):
        super().__init__(obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, 
                         log_std_max, num_layers, num_filters, device, robot_shape, cnn_stride, cnn_3dconv, robot_encoder_architecture, cost)

        decoder = m.Decoder(num_channels = obs_shape[0], 
                            feature_dim = encoder_feature_dim, 
                            num_layers = num_layers,
                            num_filters = num_filters)

        self.autoencoder = m.AutoEncoder(self.critic.encoder, decoder, robot_shape > 0).to(device)


class ATC_Model(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, log_std_min, 
                log_std_max, num_layers, num_filters, device, atc_encoder_feature_dim, atc_hidden_feature_dim):
        super().__init__()

        shared_cnn = m.ATCSharedCNN(obs_shape = obs_shape,
                                 num_layers = num_layers,
                                 num_filters = num_filters)

        actor_encoder = m.Encoder(cnn = shared_cnn,
                                  projection= m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))
        
        critic_encoder = m.Encoder(cnn = shared_cnn,
                                   projection = m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))

        critic_encoder_target = m.Encoder(cnn = m.ATCSharedCNN(obs_shape = obs_shape, num_layers = num_layers, num_filters = num_filters),
                                          projection = m.RLProjection(shared_cnn.out_dim, encoder_feature_dim))


        self.actor = m.Actor(encoder = actor_encoder,
                             action_dim = action_shape[0],
                             hidden_dim = hidden_dim,
                             log_std_min = log_std_min,
                             log_std_max = log_std_max).to(device)
        
        self.critic = m.Critic(encoder = critic_encoder,
                               action_dim = action_shape[0],
                               hidden_dim = hidden_dim).to(device)
        
        self.critic_target = m.Critic(encoder = critic_encoder_target,
                                      action_dim = action_shape[0],
                                      hidden_dim = hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        
        atc_encoder = m.Encoder(cnn = shared_cnn,
                                projection= m.PlainProjection(shared_cnn.out_dim, atc_encoder_feature_dim))    
        self.atc = m.ATC(atc_encoder, atc_hidden_feature_dim).to(device)
        
        self.atc_encoder_target = m.Encoder(cnn = m.ATCSharedCNN(obs_shape = obs_shape,
                                                                 num_layers = num_layers,
                                                                 num_filters = num_filters),
                                            projection= m.PlainProjection(shared_cnn.out_dim, atc_encoder_feature_dim)).to(device)   
        self.atc_encoder_target.load_state_dict(atc_encoder.state_dict())
    

    def soft_update_params(self, critic_tau, encoder_tau):
        for param, target_param in zip(self.critic.Q1.parameters(), self.critic_target.Q1.parameters()):
            target_param.data.copy_(critic_tau * param.data + (1 - critic_tau) * target_param.data) 
            
        for param, target_param in zip(self.critic.Q2.parameters(), self.critic_target.Q2.parameters()):
            target_param.data.copy_(critic_tau * param.data + (1 - critic_tau) * target_param.data)   

        for param, target_param in zip(self.critic.encoder.parameters(), self.critic_target.encoder.parameters()):
            target_param.data.copy_(encoder_tau * param.data + (1 - encoder_tau) * target_param.data)   


    def soft_update_params_atc(self, atc_encoder_tau):
        for param, target_param in zip(self.atc.encoder.parameters(), self.atc_encoder_target.parameters()):
            target_param.data.copy_(atc_encoder_tau * param.data + (1 - atc_encoder_tau) * target_param.data)   
