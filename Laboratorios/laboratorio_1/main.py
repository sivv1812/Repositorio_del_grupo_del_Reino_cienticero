from animales import animales

# Se elimino la linea if __name__ == '__main__': debido a se un if que no tiene sentido si estar dentro de una función.
print("Ingrese el animalito para saber que dice:")
animal = input()
animales(animal)
