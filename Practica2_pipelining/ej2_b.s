#Fernanda Mora, Arquitectura de Computadoras

#2b. Ahora utilice un algoritmo basado en corrimientos a la izquierda para calcular las potencias de 2

.data
i: .word32 0
.text
	daddi r1,r0,0;
	daddi r2,r0,10;
	daddi r3,r0,0;
	daddi r4,r0,2;
	daddi r5,r0,1;
LOOP: daddi r1,r1,1
	dsll r5,r5,1;
	sw r5,0(r3)
	daddi r3,r3,8;
	bne r1,r2,LOOP;
ENDW:	nop
	halt
