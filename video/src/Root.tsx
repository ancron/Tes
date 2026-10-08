import { Composition } from "remotion";
import { DURATION, RoseAsh } from "./RoseAsh";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition id="RoseAsh" component={RoseAsh} durationInFrames={DURATION} fps={30} width={1920} height={1080} />
    </>
  );
};
