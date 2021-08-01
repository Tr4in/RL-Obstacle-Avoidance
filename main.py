import unreal
from environment import Environment
from agent import Agent

#img_path = 'Game\Screenshots\game_scene.png'
#screenshot = unreal.AutomationLibrary()
#agentCamera = unreal.GameplayStatics.get_all_actors_of_class_with_tag(unreal.EditorLevelLibrary.get_editor_world(), unreal.CameraActor, 'agentCamera')[0]
#unreal.log(agentCamera)
#screenshot.take_high_res_screenshot(224, 224, img_path, None)

import threading

n_laser = 24
environment = Environment(n_laser)
unreal_agent = unreal.GameplayStatics.get_all_actors_with_tag(unreal.EditorLevelLibrary.get_game_world(), 'agent')[0]
speed = 100
agent = Agent(unreal_agent, speed, 0.99, 0.75, 0.2, n_laser, 10, 200, 3, 3)
NUM_EPISODES = 10
NUM_ITERATIONS = 100


def actor_hit(self_actor, other_actor, normal_impulse, hit):
    unreal.log('Hit')

actor_hit_signature = unreal.ActorHitSignature()
actor_hit_signature.add_callable(actor_hit)

unreal_agent.set_editor_property('on_actor_hit', actor_hit_signature)


def start_training():
    for iteration in range(NUM_ITERATIONS):
        create_replay_memory()
        agent.compute_learning_target()
        agent.optimize_q_network()

        if iteration == 1500:
            agent.swap_models()
    
    #for episode in range(NUM_EPISODES):
    #    action = agent.get_next_action(observation)
    #    reward, next_state, done = environment.step(agent, action, False)
    #    agent.learn()


def create_replay_memory():
    state = environment.reset()
    agent.experience_replay_counter = 0

    for episode in range(NUM_EPISODES):
        done = False
        unreal.log('Replay Episode {}'.format(episode))
        while not done and agent.experience_replay_counter < agent.experience_memory_size: # export in one agent function
            action = agent.get_next_action(state)
            reward, next_state, done = environment.step(agent.unreal_agent, action)
            #unreal.log(next_state)
            agent.store_transition(state, action, reward, next_state)
            state = next_state


threading.Thread(target = start_training).start()

#speed = 100
#agent2 = unreal.GameplayStatics.get_all_actors_with_tag(unreal.EditorLevelLibrary.get_game_world(), 'agent')[0]
#camera = unreal.GameplayStatics.get_all_actors_of_class(unreal.EditorLevelLibrary.get_game_world(), unreal.SceneCapture2D)[0]
#unreal.log(camera.get_editor_property('capture_component2d').get_editor_property('texture_target'))

#n_laser_beam = 20
#agent = Agent(0.99, 0.75, 0.2, n_laser_beam, 255, 3)
#agent.learn()

#environment = Environment()
#environment.step(agent, 1)

#renderTarget = camera.get_editor_property('capture_component2d').get_editor_property('texture_target')
#renderTarget.export_to_disk(r'C:\Users\Bushw\OneDrive\Dokumente\Unreal Projects\DepthEstimation\Saved\Screenshots\Windows\Game\Screenshots\game', unreal.ImageWriteOptions(unreal.DesiredImageFormat.PNG, async_= False))


#agent.set_actor_location(agent.get_actor_location() + agent.get_actor_forward_vector() * speed, True, True)

#unreal.log(agent.get_actor_forward_vector())