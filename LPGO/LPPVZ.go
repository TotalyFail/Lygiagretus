package main

import "fmt"

/*func main() {
	var sendingChannel = make(chan int, 300)

	for i := 0; i < 5; i++ {
		go sendData(sendingChannel, i)
	}

	fmt.Println(sendingChannel)
}*/

func sendData(sendingChannel chan<- int, multiplier int) {
	multiplier++
	var rez = 0
	for i := 1; i < 6; i++ {
		rez = i * multiplier
		fmt.Println(rez)
		sendingChannel <- rez
	}

}
