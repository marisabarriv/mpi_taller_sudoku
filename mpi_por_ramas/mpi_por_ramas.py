"""MPI: resolver los tableros en orden, repartiendo las ramas."""

import argparse
import sys
from pathlib import Path

from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sudoku import puede_colocar, resolver


def leer_seeds(ruta):
    """Lee los tableros del archivo seeds.txt."""
    seeds = {}

    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()

        if not linea or linea.startswith("#"):
            continue

        identificador, contenido = linea.split("|", 1)
        valores = [
            int(c) for c in contenido.strip().replace(".", "0")
        ]

        if len(valores) != 81:
            raise ValueError(
                f"El tablero {identificador} no tiene 81 caracteres."
            )

        seeds[identificador.strip()] = [
            valores[i:i + 9] for i in range(0, 81, 9)
        ]

    return seeds


def generar_ramas(tablero):
    """Genera las posibles primeras ramas del tablero."""
    for fila in range(9):
        for columna in range(9):
            if tablero[fila][columna] == 0:
                ramas = []

                for valor in range(1, 10):
                    if puede_colocar(tablero, fila, columna, valor):
                        copia = [f[:] for f in tablero]
                        copia[fila][columna] = valor
                        ramas.append(copia)

                return ramas

    return []


def resolver_rama(rama):
    """Resuelve una rama de búsqueda de manera independiente."""
    copia = [fila[:] for fila in rama]

    if resolver(copia):
        return copia

    return None


def resolver_tablero(comm, tablero_original):
    """Distribuye las ramas de un tablero entre los procesos MPI."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        ramas = generar_ramas(tablero_original)

        if not ramas:
            copia = [fila[:] for fila in tablero_original]
            resultado = copia if resolver(copia) else None
            return resultado

        tareas = [
            (indice, rama)
            for indice, rama in enumerate(ramas)
        ]

        # El coordinador también puede resolver ramas.
        asignaciones = [[] for _ in range(size)]

        for indice, rama in tareas:
            asignaciones[indice % size].append((indice, rama))

    else:
        asignaciones = None

    asignaciones = comm.scatter(asignaciones, root=0)

    solucion_local = None

    for indice, rama in asignaciones:
        resultado = resolver_rama(rama)

        if resultado is not None and solucion_local is None:
            solucion_local = (indice, resultado)

    soluciones = comm.gather(solucion_local, root=0)

    if rank == 0:
        soluciones_validas = [
            solucion for solucion in soluciones
            if solucion is not None
        ]

        if soluciones_validas:
            soluciones_validas.sort(key=lambda elemento: elemento[0])
            return soluciones_validas[0][1]

        return None

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Resolver Sudokus repartiendo ramas con MPI."
    )

    parser.add_argument(
        "--archivo",
        type=Path,
        default=ROOT / "seeds.txt"
    )

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Error: se necesitan al menos 2 procesos MPI.")
        return

    if rank == 0:
        try:
            ruta = args.archivo
            if not ruta.is_absolute():
                ruta = ROOT / ruta

            seeds = leer_seeds(ruta)
            error = None

        except (OSError, ValueError) as exc:
            seeds = None
            error = str(exc)

    else:
        seeds = None
        error = None

    error = comm.bcast(error, root=0)

    if error:
        if rank == 0:
            print(f"Error: {error}")
        return

    identificadores = comm.bcast(
        sorted(seeds.keys(), key=lambda x: int(x)) if rank == 0 else None,
        root=0
    )

    if rank == 0:
        print(f"Tableros a resolver: {len(identificadores)}")
        print(f"Procesos MPI: {size}")

    # Cada proceso participa en la sincronización antes de iniciar.
    comm.Barrier()

    if rank == 0:
        inicio = MPI.Wtime()

    for identificador in identificadores:
        tablero = (
            seeds[identificador]
            if rank == 0 else None
        )

        tablero = comm.bcast(tablero, root=0)

        solucion = resolver_tablero(comm, tablero)

        if rank == 0:
            if solucion is not None:
                print(f"ID: {identificador} - Resuelto")
            else:
                print(f"ID: {identificador} - Sin solución")

    comm.Barrier()

    if rank == 0:
        tiempo_total = MPI.Wtime() - inicio

        print(f"\nTableros procesados: {len(identificadores)}")
        print(f"Procesos MPI: {size}")
        print(f"Tiempo total MPI: {tiempo_total:.6f} segundos")


if __name__ == "__main__":
    main()