use hyzero::session::SessionObj;
use std::{
    sync::{Arc}
};

use tokio::{
    net::{TcpListener, TcpStream}
};

#[tokio::main]
async fn main () {

    //start listener socket
    //create interface to bind to
    //handle connection and assign to game
    //accept moves & request for game state
    //need to lock game state after turn
    //for each listener request, assign game state & vars to it & put lock on game_state
    //need to have waiting on opponent state
    //could maintain a sessionobj vec, and keep appending for new 2 connections
    //when num_players = 2, then start_session returns static sessionObj with func init_state
    create_server().await;

}

pub async fn create_server(){
    let session_obj = Arc::new(SessionObj::start_session());
    let listener = TcpListener::bind("127.0.0.1:7878").await.unwrap();
    let num_waiting = 0;
    //is it better to maintain
    loop {

        let (socket, _) = listener.accept().await.unwrap();
        tokio::spawn(
            async move {
                handle_connection(socket, num_waiting).await;

            }
        );

    }
}

async fn handle_connection (stream: TcpStream, num_waiting = i32){
    //check count
    //see if we need to make a new game_state if num_waiting = 1
    //if num_waiting == 1 game_state start and add to gamesession arr
    //when player connects need to initialize player, extern player needs to be able to accept stream as a param for intiializing

    //init player obj itself, then get the extern_player, game_State is able to call extern_player get_move
    //need to implement extern_player with tcpstream 
    if num_waiting == 1 {
        
    }

}