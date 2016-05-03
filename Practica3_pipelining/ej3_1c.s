# Fernanda Mora, Arquitectura de Computadoras
# 3.1c) Ejecute las tres versiones del código y verifique su ejecución.

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
LOOP:	lw r10,0(r1);
	lw r11,-8(r1);
	daddi r10,r10,4;
	daddi r11,r11,4;
	lw r12,-16(r1);
	lw r13,-24(r1);
	daddi r12,r12,4;
	daddi r13,r13,4;
	sw r10,0(r1);
	sw r11,-8(r1);
	sw r12,-16(r1);
	sw r13,-24(r1);
	daddi r1,r1,-32;
	bne r1,r0,LOOP;
ENDW: nop
	halt
