<?php

/******************************************************************************************
* ARLSTem: is a free Arabic light stemmer that is based on some rules for stripping affixes.
*
* K. Abainia, S. Ouamour and H. Sayoud, "A Novel Robust Arabic Light Stemmer", International
* Journal of Experimental & Theoretical Artificial Intelligence (Taylor & Francis).
* DOI:10.1080/0952813X.2016.1212100
******************************************************************************************/

class ARLSTem
{
	private $m_word; // array of utf-8 characters
	
	public function ARLSTem($str)
		{
			// split the utf-8 string into an array of characters
			$arr = array();
			$strLen = mb_strlen($str, 'UTF-8');
			for ($i = 0; $i < $strLen; $i++)
			{
				$arr[] = mb_substr($str, $i, 1, 'UTF-8');
			}
			// keep the characters array of the word
			$this->m_word = $arr;
		}
		
	public function getStem()
		{
			$this->normalize();
			$b1 = $this->deletePrefixes();
			$b2 = $this->deleteSuffixes();
			if(!$this->pluralToSingular()) if(!$this->feminineToMasculine()) if(!$b1) $this->verbStemming();
			
			return $this->getWord();
		}
		
	public function getWord()
		{
			$temp='';
			$nb_chars = count($this->m_word);
			// concatenate the characters array in a single string
			for($i=0; $i<$nb_chars; $i++) $temp .= $this->m_word[$i];
			
			return $temp;
		}
		
	private function normalize()
		{
			$len = count($this->m_word);
			for($i=0; $i<$len; $i++)
			{
				// replace Hamzated Alif with Alif bare
				if(utf8_char_equal($this->m_word[$i], 0xD8A2) || 
				   utf8_char_equal($this->m_word[$i], 0xD8A3) || 
				   utf8_char_equal($this->m_word[$i], 0xD8A5))
				{
					utf8_replace_char($this->m_word[$i], 0xA7D8);
				}
				// replace Alif MaqSura with Yaa
				if(utf8_char_equal($this->m_word[$i], 0xD989))
				{
					utf8_replace_char($this->m_word[$i], 0x8AD9);
				}
			}
			// remove the Waaw from the beginning if the remaining is 4 characters at least
			if($len >= 4 && utf8_char_equal($this->m_word[0], 0xD988))
			{
				$this->m_word = array_slice($this->m_word,1,$len);
			}
		}
	
	private function feminineToMasculine()
		{
			$len = count($this->m_word);
			// remove the taaMarbuta at the end if the remaining is 4 characters at least
			if($len >= 4 && utf8_char_equal($this->m_word[$len-1], 0xD8A9))
			{
				$this->m_word = array_slice($this->m_word,0,$len-1);
				return true;
			}
			return false;
		}
	
	private function deletePrefixes()
		{
			$len = count($this->m_word);
			// remove baa, alif and laam from the beginning if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8A8) &&
							utf8_char_equal($this->m_word[1], 0xD8A7) &&
							utf8_char_equal($this->m_word[2], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,3,$len);
				return true;
			}
			// remove kaaf, alif and laam from the beginning if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD983) &&
							utf8_char_equal($this->m_word[1], 0xD8A7) &&
							utf8_char_equal($this->m_word[2], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,3,$len);
				return true;
			}
			// remove waaw, alif and laam from the beginning if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD988) &&
							utf8_char_equal($this->m_word[1], 0xD8A7) &&
							utf8_char_equal($this->m_word[2], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,3,$len);
				return true;
			}
			// remove faa, baa, alif and laam from the beginning if the remaining is 3 characters at least
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD981) &&
							utf8_char_equal($this->m_word[1], 0xD8A8) &&
							utf8_char_equal($this->m_word[2], 0xD8A7) &&
							utf8_char_equal($this->m_word[3], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,4,$len);
				return true;
			}
			// remove waaw, baa, alif and laam from the beginning if the remaining is 3 characters at least
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD988) &&
							utf8_char_equal($this->m_word[1], 0xD8A8) &&
							utf8_char_equal($this->m_word[2], 0xD8A7) &&
							utf8_char_equal($this->m_word[3], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,4,$len);
				return true;
			}
			// remove faa, kaaf, alif and laam  from the beginning if the remaining is 3 characters at least
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD981) &&
							utf8_char_equal($this->m_word[1], 0xD983) &&
							utf8_char_equal($this->m_word[2], 0xD8A7) &&
							utf8_char_equal($this->m_word[3], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,4,$len);
				return true;
			}
			// remove faa, laam and laam from the beginning if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD981) &&
							utf8_char_equal($this->m_word[1], 0xD984) &&
							utf8_char_equal($this->m_word[2], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,3,$len);
				return true;
			}
			// remove waa, laam and laam from the beginning if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD988) &&
							utf8_char_equal($this->m_word[1], 0xD984) &&
							utf8_char_equal($this->m_word[2], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,3,$len);
				return true;
			}
			// remove alif and laam from the beginning if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A7) &&
							utf8_char_equal($this->m_word[1], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				//$this->m_word = null;
				//$this->m_word = $temp;
				return true;
			}
			// remove laam and laam from the beginning if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD984) &&
							utf8_char_equal($this->m_word[1], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			// remove faa and laam from the beginning if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD981) &&
							utf8_char_equal($this->m_word[1], 0xD984))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			// remove faa and baa from the beginning if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD981) &&
							utf8_char_equal($this->m_word[1], 0xD8A8))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			return false;
		}
		
	private function deleteSuffixes()
		{
			$len = count($this->m_word);
			// remove kaaf at the end if the remaining is 3 characters at least
			if($len >= 4 && utf8_char_equal($this->m_word[$len-1], 0xD983))
			{
				$this->m_word = array_slice($this->m_word,0,$len-1);
				return true;
			}
			// remove kaaf and yaa at the end if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD983) &&
							utf8_char_equal($this->m_word[$len-1], 0xD98A))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove kaaf and miim at the end if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD983) &&
							utf8_char_equal($this->m_word[$len-1], 0xD985))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove kaaf, miim and alif at the end if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD983) &&
							utf8_char_equal($this->m_word[$len-2], 0xD985) &&
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove kaaf, noon and shedda at the end if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD983) &&
							utf8_char_equal($this->m_word[$len-2], 0xD986) &&
							utf8_char_equal($this->m_word[$len-1], 0xD991))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove haa at the end if the remaining is 3 characters at least
			if($len >= 4 && utf8_char_equal($this->m_word[$len-1], 0xD987))
			{
				$this->m_word = array_slice($this->m_word,0,$len-1);
				return true;
			}
			// remove haa and alif at the end if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD987) &&
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove haa and miim at the end if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD987) &&
							utf8_char_equal($this->m_word[$len-1], 0xD985))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove haa, miim and alif at the end if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD987) &&
							utf8_char_equal($this->m_word[$len-2], 0xD985) &&
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove haa, noon and shedda at the end if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD987) &&
							utf8_char_equal($this->m_word[$len-2], 0xD986) &&
							utf8_char_equal($this->m_word[$len-1], 0xD991))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove noon and alif at the end if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD986) &&
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			return false;
		}
		
	private function pluralToSingular()
		{
			$len = count($this->m_word);
			// remove alif and noon if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD8A7) &&
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove yaa and noon if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD98A) &&
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove waaw and noon if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD988) &&
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove alif and taa if the remaining is 3 characters at least
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD8A7) &&
							utf8_char_equal($this->m_word[$len-1], 0xD8AA))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove taa, alif and noon at the end if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD8AA) &&
							utf8_char_equal($this->m_word[$len-2], 0xD8A7) &&
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove taa, yaa and noon at the end if the remaining is 3 characters at least
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD8AA) &&
							utf8_char_equal($this->m_word[$len-2], 0xD98A) &&
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove alif at the third position if it also exists at the first position
			if($len >= 4 && utf8_char_equal($this->m_word[0], 0xD8A7) &&
							utf8_char_equal($this->m_word[2], 0xD8A7))
			{
				$this->m_word = array_merge(array_slice($this->m_word,0,2) , array_slice($this->m_word,3,$len));
				return true;
			}
			// remove alif from the beginning and before the last char
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A7) &&
							utf8_char_equal($this->m_word[$len-2], 0xD8A7))
			{
				$this->m_word = array_merge(array_slice($this->m_word,1,$len-3) , array_slice($this->m_word,$len-1,$len));
				return true;
			}
			return false;
		}
		
	private function verbStemming()
		{
			$len = count($this->m_word);
			// remove taa from the beginning # yaa and noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove taa from the beginning # alif and noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove taa from the beginning # waaw and noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove yaa from the beginning # alif and noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-2], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove yaa from the beginning # waaw and noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove alif (hamzat wassel) from the beginning # waaw and alif from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8A5) && 
							utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove alif from the beginning # waaw and alif from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,1,$len-3);
				return true;
			}
			// remove alif (hamzat wassel) from the beginning # yaa from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A5) && 
							utf8_char_equal($this->m_word[$len-1], 0xD98A))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove alif from the beginning # yaa from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD98A))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove alif (hamzat wassel) from the beginning # alif from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A5) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove alif from the beginning # alif from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove alif (hamzat wassel) from the beginning # noon from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A5) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove alif from the beginning # noon from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove yaa from the beginning # noon from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len-2);
				return true;
			}
			// remove taa from the beginning # noon from the end
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1, $len-2);
				return true;
			}
     
			/**************************************************************************
			* future = siin + present
			**************************************************************************/
			// remove siin and taa from the beginning # yaa and noon from the end
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-3);
				return true;
			}
			// remove siin and taa from the beginning # alif and noon from the end
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-3);
				return true;
			}
			// remove siin and taa from the beginning # waaw and noon from the end
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-3);
				return true;
			}
			// remove siin and yaa from the beginning # alif and noon from the end
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-2], 0xD8A7) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-3);
				return true;
			}
			// remove siin and yaa from the beginning # waaw and noon from the end
			if($len >= 7 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-3);
				return true;
			}
			// remove siin and yaa from the beginning # noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD98A) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-2);
				return true;
			}
			// remove siin and taa from the beginning # noon from the end
			if($len >= 6 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len-2);
				return true;
			}
     
			/**************************************************************************
			* At the end
			**************************************************************************/
			// remove taa, miim and alif from the end
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD985) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-4);
				return true;
			}
			// remove taa, noon and chedda from the end
			if($len >= 6 && utf8_char_equal($this->m_word[$len-3], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-2], 0xD986) && 
							utf8_char_equal($this->m_word[$len-1], 0xD991))
			{
				$this->m_word = array_slice($this->m_word,0,$len-4);
				return true;
			}
			// remove noon and alif from the end
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD986) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove taa and miim from the end
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-1], 0xD985))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove taa and alif from the end
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD8AA) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove waaw and alif from the end
			if($len >= 5 && utf8_char_equal($this->m_word[$len-2], 0xD988) && 
							utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-3);
				return true;
			}
			// remove taa the end
			if($len >= 4 && utf8_char_equal($this->m_word[$len-1], 0xD8AA))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove alif from the end
			if($len >= 4 && utf8_char_equal($this->m_word[$len-1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
			// remove noon from the end
			if($len >= 4 && utf8_char_equal($this->m_word[$len-1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,0,$len-2);
				return true;
			}
     
			/**************************************************************************
			* At the begining
			**************************************************************************/
			// remove alif from the beginning
			if($len >= 4 && utf8_char_equal($this->m_word[0], 0xD8A7) || 
							utf8_char_equal($this->m_word[0], 0xD8A3)  || 
							utf8_char_equal($this->m_word[0], 0xD8A5))
			{
				$this->m_word = array_slice($this->m_word,1,$len);
				return true;
			}
			// remove noon from the beginning
			if($len >= 4 && utf8_char_equal($this->m_word[0], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,1,$len);
				return true;
			}
			// remove taa from the beginning
			if($len >= 4 && utf8_char_equal($this->m_word[0], 0xD8AA))
			{
				$this->m_word = array_slice($this->m_word,1,$len);
				return true;
			}
			// remove yaa from the beginning
			if($len >= 4 && utf8_char_equal($this->m_word[0], 0xD98A))
			{
				$this->m_word = array_slice($this->m_word,1,$len);
				return true;
			}
     
			/**************************************************************************
			* Futur = siin + present
			**************************************************************************/
			// remove siin and alif from the beginning
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD8A7))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			// remove siin and noon from the beginning
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD986))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			// remove siin and taa from the beginning
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD8AA))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			// remove siin and yaa from the beginning
			if($len >= 5 && utf8_char_equal($this->m_word[0], 0xD8B3) && 
							utf8_char_equal($this->m_word[1], 0xD98A))
			{
				$this->m_word = array_slice($this->m_word,2,$len);
				return true;
			}
			return false;
		}
}

// check the equivalence of two characters
function utf8_char_equal($char, $hex)
{
	if(strlen($char) == 1) return false; // if the utf-8 character is not Arabic
	return (((ord($char[0])<<8) | ord($char[1])) == $hex);
}

// replace an utf-8 character with another (given the hex code point)
function utf8_replace_char(&$char, $hex)
{
	$char[0] = chr($hex & 0xFF);
	$char[1] = chr($hex>>8);
}

?>