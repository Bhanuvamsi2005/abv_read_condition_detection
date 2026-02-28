import pygame
import os


class Alarm:
    def __init__(self, sound_file="alarm.wav"):

        pygame.mixer.init()

        self.sound_path = os.path.join(os.getcwd(), sound_file)

        if not os.path.exists(self.sound_path):
            print("❌ ERROR: alarm.wav NOT FOUND")
            print(f"Expected at: {self.sound_path}")
            self.sound = None
        else:
            print(f"✅ Alarm sound loaded: {self.sound_path}")
            self.sound = pygame.mixer.Sound(self.sound_path)

        self.playing = False

    def start(self):
        if self.sound and not self.playing:
            self.sound.play(-1)
            self.playing = True
            print("🚨 ALARM STARTED")

    def stop(self):
        if self.sound and self.playing:
            self.sound.stop()
            self.playing = False
            print("✅ Alarm stopped")