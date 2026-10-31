
"""MPI: distribuir Sudokus independientes entre procesos.

El proceso 0 coordina las tareas y reúne los resultados.
Los demás procesos resuelven tableros mediante backtracking.
"""

import argparse
import sys
import time
from pathlib import Path

from mpi4py import MPI


# Directorio principal del proyecto.
ROOT = Path(__file__).resolve().parent.parent

# Permite importar sudoku.py desde el directorio principal.
sys.path.insert(0, str(ROOT))

from sudoku import resolver_con_tiempo


TAG_TAREA = 1
TAG_RESULTADO = 2


def leer_seeds(ruta):
    """Lee las semillas con formato id|tablero."""
    seeds = {}

    for texto in ruta.read_text(encoding="utf-8").splitlines():
        texto = texto.strip()

        if not texto or texto.startswith("#"):
            continue

        identificador, contenido = texto.split("|", 1)
        celdas = [
            int(c) for c in contenido.strip().replace(".", "0")
        ]

        if len(celdas) != 81:
            raise ValueError(
                f"El tablero {identificador} no tiene 81 caracteres."
            )

        seeds[identificador.strip()] = [
            celdas[i:i + 9] for i in range(0, 81, 9)
        ]

    return seeds


def resolver_tarea(tarea):
    """Resuelve una tarea y devuelve su resultado."""
    identificador, original = tarea

    # Cada trabajador recibe una copia independiente.
    tablero = [fila[:] for fila in original]

    resuelto, segundos = resolver_con_tiempo(tablero)

    return identificador, resuelto, segundos


def ejecutar_maestro(comm, tareas):
    """Distribuye tareas y reúne los resultados en el proceso 0."""
    cantidad_procesos = comm.Get_size()
    cantidad_trabajadores = cantidad_procesos - 1

    resultados = []
    siguiente_tarea = 0

    # El tiempo empieza antes de distribuir el primer tablero.
    inicio = MPI.Wtime()

    # Entregar una tarea inicial a cada trabajador disponible.
    for trabajador in range(1, cantidad_procesos):
        if siguiente_tarea < len(tareas):
            comm.send(
                tareas[siguiente_tarea],
                dest=trabajador,
                tag=TAG_TAREA
            )
            siguiente_tarea += 1
        else:
            comm.send(None, dest=trabajador, tag=TAG_TAREA)

    # Recibir resultados y asignar nuevas tareas dinámicamente.
    while len(resultados) < len(tareas):
        estado = MPI.Status()

        resultado = comm.recv(
            source=MPI.ANY_SOURCE,
            tag=TAG_RESULTADO,
            status=estado
        )

        trabajador = estado.Get_source()
        resultados.append(resultado)

        # Si quedan tableros, asignar el siguiente.
        if siguiente_tarea < len(tareas):
            comm.send(
                tareas[siguiente_tarea],
                dest=trabajador,
                tag=TAG_TAREA
            )
            siguiente_tarea += 1
        else:
            # No quedan tareas: indicar al trabajador que termine.
            comm.send(None, dest=trabajador, tag=TAG_TAREA)

    tiempo_total = MPI.Wtime() - inicio

    # Mostrar resultados ordenados por ID.
    resultados.sort(key=lambda resultado: resultado[0])

    for identificador, resuelto, segundos in resultados:
        print(f"\nID: {identificador}")
        print("Resuelto" if resuelto else "Sin solución")
        print(f"Tiempo parcial: {segundos:.6f} segundos")

    print(f"\nTableros procesados: {len(resultados)}")
    print(f"Trabajadores MPI: {cantidad_trabajadores}")
    print(f"Tiempo total MPI: {tiempo_total:.6f} segundos")


def ejecutar_trabajador(comm):
    """Recibe y resuelve tareas hasta recibir la señal de finalización."""
    while True:
        tarea = comm.recv(source=0, tag=TAG_TAREA)

        if tarea is None:
            break

        resultado = resolver_tarea(tarea)

        comm.send(
            resultado,
            dest=0,
            tag=TAG_RESULTADO
        )


def main():
    parser = argparse.ArgumentParser(
        description="Resolver Sudokus distribuyendo tableros con MPI."
    )

    parser.add_argument(
        "--archivo",
        type=Path,
        default=ROOT / "seeds.txt",
        help="Archivo de semillas."
    )

    seleccion = parser.add_mutually_exclusive_group()

    seleccion.add_argument(
        "--id",
        help="Resolver solamente el tablero indicado."
    )

    seleccion.add_argument(
        "--all",
        action="store_true",
        help="Resolver todos los tableros."
    )

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Solamente el coordinador lee el archivo.
    if rank == 0:
        try:
            if size < 2:
                raise ValueError(
                    "Se necesita al menos 2 procesos MPI."
                )

            ruta = args.archivo
            if not ruta.is_absolute():
                ruta = ROOT / ruta

            seeds = leer_seeds(ruta)

            if not seeds:
                raise ValueError("No se encontraron semillas.")

            if args.id:
                if args.id not in seeds:
                    raise ValueError(
                        f"El ID {args.id} no existe en el archivo."
                    )

                tareas = [(args.id, seeds[args.id])]
            else:
                tareas = list(seeds.items())

            error = None

        except (OSError, ValueError) as exc:
            tareas = None
            error = str(exc)

    else:
        tareas = None
        error = None

    # Compartir la validación y las tareas con todos los procesos.
    error, tareas = comm.bcast(
        (error, tareas),
        root=0
    )

    if error:
        if rank == 0:
            print(f"Error: {error}")
        return

    if rank == 0:
        print(f"Tableros a resolver: {len(tareas)}")
        print(f"Procesos MPI: {size}")
        ejecutar_maestro(comm, tareas)

    else:
        ejecutar_trabajador(comm)


if __name__ == "__main__":
    main()
