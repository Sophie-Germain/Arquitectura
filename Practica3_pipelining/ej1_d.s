#Fernanda Mora, Arquitectura de Computadoras

#1d) Modifique el programa anterior para que las variables y el código se almacenen a partir de las direcciones 100 y 200 de sus  respectivos segmentos de datos y código

.data
.org 100
i:	.word32 0
j:	.word32 0
.text
.org 200
	daddi R2,R0,0;
	daddi r3, R0, 0;
	daddi r5,R0,10 ;
WHIL:	slt R6, R2, R5
	beqz R6, ENDW
	daddi r3, r3, 5
	sw R3, j(r0)
	daddi r2,r2,1
	sw	r2,i(r0)
	j WHIL
ENDW:	nop
	halt
