from critics import *
from generators import *
from translator import *
from data import *
from cycleGAN import *

generatorAB = ConvGenerator("GeneratorAB")
generatorBA = ConvGenerator("GeneratorBA")

criticA = FCCritic(name="CriticA")
criticB = FCCritic(name="CriticB")

translatorA = Translator(generatorAB, generatorBA, criticB, name="A")
translatorB = Translator(generatorBA, generatorAB, criticA, name="B", reuse=True)

horse = Animal("trainA")
zebra = Animal("trainB")

batch_size = 8
steps = 100000

cycleGAN = CycleGAN([translatorA, translatorB], [horse, zebra])
cycleGAN(batch_size, steps, project_path.model_path)
