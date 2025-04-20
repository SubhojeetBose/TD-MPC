import torch
import torch.nn.functional as F

import utils
from agent.networks.actor import SacActor
from agent.networks.critic import SacCritic


class Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
    ):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.actor_repr_dim = obs_shape[0]
        self.critic_repr_dim = obs_shape[0]+action_shape[0]

        # TODO: Define an actor network
        self.actor = SacActor(self.actor_repr_dim, action_shape, hidden_dim).to(device)

        # TODO: Define a critic network and a target critic network
        self.critic = SacCritic(self.critic_repr_dim, action_shape, hidden_dim).to(device)
        self.critic_target = SacCritic(self.critic_repr_dim, action_shape, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # TODO: Define the optimizers for the actor and critic networks
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # as given in the paper we will maintain a temperature parameter and adaptively learn it
        # used this https://github.com/denisyarats/pytorch_sac/blob/master/agent/sac.py link to understand how to do soft updates
        # the idea is basically to adaptively learn this temperature as well, paper states if not learnt adaptively it affects the performance
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha).item()
        self.target_entropy = -action_shape[0]
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.train()
        self.critic_target.train()

    def __repr__(self):
        return "rl"

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)

        # TODO: Get the action distribution from the actor network
        dist = self.actor(obs.unsqueeze(0))

        if eval_mode:
            # TODO: Sample an action from the distribution in eval mode
            action = dist.sample(clip=1.)[0]
        else:
            # If step is less than the number of exploration steps, sample a random action.
            # Otherwise, sample an action from the distribution.
            if step < self.num_expl_steps:
                # TODO: Sample a random action between -1 and 1
                action = (-2 * torch.rand(dist.mean.shape) + 1)[0] # Get random b/w [-1, 1]
            else:
                # TODO: Sample an action from the distribution
                action = dist.sample(clip=1.)[0]

        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            # TODO: Compute the target Q value
            # Hint: Use next obs and next action to compute the target Q value
            dist = self.actor(next_obs.unsqueeze(0))     #using actor to get the dist of next action, here std dev is calculated inside the actor itself
            # tanh gives bad result, instead had to clip the values to 1.0
            next_action = dist.sample(clip=1.)[0]
            target_Q = reward + discount * self.critic_target(next_obs, next_action)

        # TODO: Compute the Q value from the critic network
        Q = self.critic(obs, action)

        # TODO: Compute the critic loss
        critic_loss = F.mse_loss(Q, target_Q)

        # TODO: Optimize the critic network
        # critic gradient descent step
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()


        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()


        # TODO: Get the action distribution from the actor network
        # and sample an action from the distribution
        # get the action from distribution
        dist = self.actor(obs.unsqueeze(0))
        action = dist.sample(clip=1.)[0]
        
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # TODO: Get the Q value from the critic network
        Q = self.critic(obs, action)

        # TODO: Compute the actor loss
        # update the actor loss using the alpha value
        actor_loss = (self.alpha * log_prob - Q).mean()

        # TODO: Optimize the actor network
        # actor gradient descent
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # temperature update
        alpha_loss = -(torch.exp(self.log_alpha) * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # predict new temperature
        self.alpha = torch.exp(self.log_alpha).item()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # convert to float
        obs = obs.float()
        next_obs = next_obs.float()
        action, reward, discount = action.float(), reward.float(), discount.float()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def save_snapshot(self):
        keys_to_save = ["actor", "critic"]
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        return payload
