import llama3Tokenizer from 'llama3-tokenizer-js'

export class Tokenizer {  
    encode(str: string) {
        return llama3Tokenizer.encode(str);
    }
  
    decode(arr: number[]) {
        return llama3Tokenizer.decode(arr);
    }
  }
