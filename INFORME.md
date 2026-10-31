# Informe de resultados

## Entorno

- Equipo: DESKTOP-8VLM9VC
- Procesador: Intel Core i5-1135G7 de 11.ª generación (2.40 GHz)
- Memoria RAM: 16 GB
- Sistema operativo: Windows de 64 bits
- Versión del contenedor: `augustosalazar/slim-mpi:2`
- Cantidad de repeticiones por medición: 3
- Valor común de `N` elegido: 8

## Mediciones realizadas

Se realizaron tres ejecuciones para cada implementación, utilizando los mismos 30 tableros del archivo `seeds.txt`. Para las estrategias MPI se mantuvo el mismo valor de `N = 8`.

| Implementación | Repetición 1 (s) | Repetición 2 (s) | Repetición 3 (s) | Mediana (s) |
|---|---:|---:|---:|---:|
| Secuencial | 95.528590 | 84.652277 | 85.368190 | 85.368190 |
| MPI por tableros | 35.464866 | 38.403170 | 34.873373 | 35.464866 |
| MPI por ramas | 153.626873 | 160.647597 | 152.661399 | 153.626873 |
| MPI híbrido | 85.695739 | 80.652301 | 83.439369 | 83.439369 |

## Comparación final con el mismo N

| Implementación | Caso | Procesos | Mediana del tiempo total (s) | Aceleración frente a secuencial |
|---|---|---:|---:|---:|
| Secuencial | Todos los IDs | 1 | 85.368190 | 1,00 |
| MPI por tableros | Todos los IDs | 8 | 35.464866 | 2,41 |
| MPI por ramas | Todos los IDs, uno después de otro | 8 | 153.626873 | 0,56 |
| MPI híbrido | Todos los IDs | 8 | 83.439369 | 1,02 |

La aceleración se calculó dividiendo la mediana del tiempo secuencial entre la mediana del tiempo de cada implementación MPI.

## Conclusiones

- La estrategia MPI por tableros obtuvo el menor tiempo para resolver los 30 tableros. Su mediana fue de 35.464866 segundos, frente a los 85.368190 segundos de la versión secuencial. Esto representa una aceleración aproximada de 2,41 veces y una reducción del tiempo cercana al 58,5%.

- La comunicación entre procesos influye en el rendimiento porque repartir el trabajo también requiere enviar y recibir información. Además, los tableros no tienen necesariamente la misma dificultad, por lo que algunos pueden tardar más en resolverse que otros.

- Al repartir los tableros, los procesos pueden trabajar en diferentes problemas al mismo tiempo, lo que permitió obtener el mejor resultado. En cambio, la estrategia por ramas tuvo un tiempo mayor que la versión secuencial, posiblemente debido al costo de comunicación y sincronización que implica repartir las ramas de cada tablero. La estrategia híbrida consiguió un tiempo ligeramente menor que el secuencial, pero no superó el rendimiento de MPI por tableros.

En general, los resultados muestran que la forma de distribuir el trabajo entre los procesos MPI influye bastante en el rendimiento. Para este conjunto de tableros, repartir los problemas completos fue la estrategia más conveniente.