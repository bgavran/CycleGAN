from critics import *
from generators import *
from translator import *
from data import *
from cycleGAN import *


batch_size = 16
steps = 1000000
img_size = 32

horse = Animal("trainA", img_size=img_size)
zebra = Animal("trainB", img_size=img_size)

generatorAB = FCGenerator(name="GeneratorAB", img_size=img_size)
generatorBA = FCGenerator(name="GeneratorBA", img_size=img_size)

criticA = FCCritic(name="CriticA", img_size=img_size)
criticB = FCCritic(name="CriticB", img_size=img_size)

translatorA = Translator(generatorAB, generatorBA, criticB, name="A")
translatorB = Translator(generatorBA, generatorAB, criticA, name="B", reuse=True)

cycleGAN = CycleGAN([translatorA, translatorB], [horse, zebra])
cycleGAN(batch_size, steps, project_path.model_path)
