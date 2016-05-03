# Fernanda Mora, Arquitectura de Computadoras
# 3.1a) Ejecute las tres versiones del código y verifique su ejecución.

.data
x0: .word32 0
x1: .word32 1
x2: .word32 2
x3: .word32 3
x4: .word32 4
x5: .word32 5
x6: .word32 6
x7: .word32 7
x8: .word32 8
x9: .word32 9
x10: .word32 10
x11: .word32 11
x12: .word32 12
x13: .word32 13
x14: .word32 14
x15: .word32 15
.text
	daddi r1,r0,128;
LOOP: lw r10,0(r1); Leer elemento de un vector
	daddi r10,r10,4 ; Sumar 4 al elemento
	sw r10,0(r1); Escribir el nuevo valor
	daddi r1,r1,-8 ; Actualizar la var. indice
	bne r1,r0,LOOP ; Fin de vector?
ENDW: nop
	halt
