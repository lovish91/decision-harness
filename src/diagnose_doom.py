import vizdoom as vzd

game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/basic.cfg")
game.set_window_visible(False)
game.set_episode_timeout(200)
game.init()

try:
    buttons = game.get_available_buttons()
    print("Timeout:", game.get_episode_timeout())

    for step in range(200):
        if game.is_episode_finished():
            print("Already finished before action", step + 1)
            break

        state = game.get_state()
        if state is None:
            print("State unavailable before action", step + 1)
            break

        button = buttons[step % len(buttons)]
        action = [button == b for b in buttons]
        reward = game.make_action(action, 1)

        print(
            f"step={step + 1}, action={button.name}, "
            f"reward={reward}, dead={game.is_player_dead()}, "
            f"finished={game.is_episode_finished()}"
        )

        if game.is_episode_finished():
            print("Episode ended after action", step + 1)
            break
finally:
    game.close()
